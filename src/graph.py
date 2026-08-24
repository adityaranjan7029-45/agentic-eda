"""
LangGraph wiring for Agentic EDA.

Architecture: a single Planner Agent (LLM) reads the dataset profile and
decides which preprocessing steps are needed and in what order. Every step
after that is executed by the existing hand-written, deterministic pandas
functions in src/nodes.py -- the LLM never writes or executes code, it only
chooses and sequences steps from a fixed, known-safe vocabulary.

Flow:
    planner -> (conditional routing on plan.steps) -> step node -> ... -> END
"""

import builtins
import io
import os
import re
import signal
import time
from groq import RateLimitError
import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- we only ever save figures, never display them
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from langgraph.graph import StateGraph, END

from src.config import get_llm
from src.nodes import (
    GraphState,
    PreprocessingPlan,
    normalize_column_name,
    data_cleaning_node,
    type_conversion_node,
    imputation_node,
    outlier_handling_node,
    feature_engineering_node,
    encoding_node,
    feature_transformation_node,
    scaling_node,
    dimensionality_reduction_node,
    feature_selection_node,
)
from src.schemas import InsightReport, VisualizationPlan, CriticVerdict

# Which Groq model each agent uses. Kept as one dict so model choices are
# visible/tunable in one place instead of scattered through node functions.
#
# NOTE: llama-3.3-70b-versatile and llama-3.1-8b-instant were both shut down
# by Groq on 2026-08-16 -- openai/gpt-oss-120b and openai/gpt-oss-20b are
# Groq's recommended replacements (console.groq.com/docs/deprecations).
#
# Both models have IDENTICAL free-tier limits (8K tokens/minute each) --
# but they're tracked as SEPARATE per-model buckets. A full pipeline run
# makes 5+ LLM calls back to back (more if a chart needs fixing or the
# Critic rejects a draft), which can exceed a single model's 8K TPM bucket.
# Splitting the lighter-weight agents (Planner, Synthesis) onto gpt-oss-20b
# means they draw from a separate budget than the heavier reasoning agents
# (Insight, Visualization, Critic) on gpt-oss-120b -- roughly doubling the
# total headroom available to one run.
AGENT_MODELS = {
    "planner": "openai/gpt-oss-20b",
    "insight": "openai/gpt-oss-120b",
    "synthesis": "openai/gpt-oss-20b",
    "visualization": "openai/gpt-oss-120b",
    "critic": "openai/gpt-oss-120b",
}

CHARTS_DIR = "charts"
MAX_CHART_FIX_ATTEMPTS = 2  # how many times we let the LLM try to fix its own broken chart code
MAX_CRITIC_REVISIONS = 2  # how many times the Critic can send the report back before we ship it anyway

# Retry settings for Groq rate limits specifically. ChatGroq already retries
# transient errors twice by default (max_retries=2 on the client), but those
# retries fire almost immediately -- fine for a network blip, not enough for
# a genuine tokens-per-minute shortfall, which needs real seconds to clear
# from Groq's rolling 60s window. This adds a second, slower retry layer
# specifically for RateLimitError (HTTP 429).
RATE_LIMIT_MAX_ATTEMPTS = 4
RATE_LIMIT_BASE_WAIT_SECONDS = 15  # attempt 1 waits 15s, attempt 2 waits 30s, attempt 3 waits 45s...


def invoke_with_retry(runnable, prompt):
    """Wraps a .invoke() call (on either a plain LLM or a
    with_structured_output-wrapped one) with a real wait-and-retry loop for
    Groq's RateLimitError. Every direct LLM call in this file goes through
    this instead of calling .invoke() directly."""
    last_error: Exception = RuntimeError("invoke_with_retry failed")
    for attempt in range(RATE_LIMIT_MAX_ATTEMPTS):
        try:
            return runnable.invoke(prompt)
        except RateLimitError as e:
            last_error = e
            if attempt == RATE_LIMIT_MAX_ATTEMPTS - 1:
                break
            wait = RATE_LIMIT_BASE_WAIT_SECONDS * (attempt + 1)
            print(f"   [!] Rate limited -- waiting {wait}s before retry ({attempt + 1}/{RATE_LIMIT_MAX_ATTEMPTS})...")
            time.sleep(wait)
    if last_error is not None:
        raise last_error
    raise RuntimeError("invoke_with_retry failed: no attempts were made.")

# Must exactly match the step names described in PreprocessingPlan (src/nodes.py)
# and the node names registered in build_graph() below.
VALID_STEPS = [
    "data_cleaning",
    "type_conversion",
    "imputation",
    "outlier_handling",
    "feature_engineering",
    "encoding",
    "feature_transformation",
    "scaling",
    "dimensionality_reduction",
    "feature_selection",
]


def build_profile(df: pd.DataFrame) -> str:
    """Builds the same style of profile string profile_dataframe() produces,
    but reads directly off the graph state's df instead of the module-level
    global used by the @tool version in nodes.py."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    null_counts = df.isnull().sum()
    null_summary = null_counts[null_counts > 0]
    info_str += f"\n\nNull values:\n{null_summary.to_string() if len(null_summary) > 0 else 'None'}"

    info_str += "\n\nSample Data (First 5 rows):\n"
    info_str += df.head().to_string()

    return info_str


def planner_node(state: GraphState):
    """The one LLM call in this graph. Looks at the dataset profile and
    decides which of VALID_STEPS apply, and in what order."""
    print("-> Planner Agent: Analyzing dataset and drafting plan...")

    df = state["df"]
    profile = build_profile(df)

    llm = get_llm(model=AGENT_MODELS["planner"])
    structured_llm = llm.with_structured_output(PreprocessingPlan)

    prompt = f"""You are a senior data scientist planning an EDA/preprocessing pipeline
for the dataset profiled below.

Choose the ordered sequence of preprocessing steps this specific dataset needs.
Do NOT include a step just because it exists -- only include steps that this
dataset's profile actually justifies (e.g. skip 'imputation' if there are no
nulls, skip 'dimensionality_reduction' if there aren't many columns).

Valid step names (use these exact strings, nothing else): {", ".join(VALID_STEPS)}

Dataset profile:
{profile}
"""

    plan = invoke_with_retry(structured_llm, prompt)

    # Safety net: silently drop any step name the LLM invents that isn't in
    # our fixed vocabulary, so a hallucinated step can't crash the router.
    plan.steps = [s for s in plan.steps if s in VALID_STEPS]

    print(f"   [+] Plan: {plan.steps}")
    print(f"   [+] Reasoning: {plan.reasoning}")

    # --- Resolve the target column ---
    # There's exactly one source of truth here: state["target_col"], set by
    # the Streamlit dropdown (or left unset if the user didn't pick one --
    # deliberately NOT auto-guessed by an LLM; that would be a second,
    # redundant detection path doing the same job the UI already does).
    # The raw name is checked against the RAW dataframe's columns (this node
    # runs before any cleaning), then normalized through the exact same
    # function data_cleaning_node uses to rename columns -- so the value we
    # store still matches once the dataframe has actually been cleaned, even
    # if the casing/spelling the user picked doesn't match post-cleanup.
    raw_target = state.get("target_col")
    target_col = normalize_column_name(raw_target) if raw_target and raw_target in df.columns else None

    if target_col:
        print(f"   [+] Target column: '{target_col}'")
    else:
        print("   [+] No target column selected -- running in general profiling mode.")

    # Freeze a copy of the chosen steps now, before any node starts popping
    # them off plan.steps -- otherwise by the time the Synthesis Agent runs,
    # plan.steps is empty and we've lost the record of what actually happened.
    return {"plan": plan, "steps_taken": list(plan.steps), "target_col": target_col}


def _find_target_columns(df: pd.DataFrame, target_col: str):
    """Resolves target_col against the ACTUAL columns present in the (fully
    preprocessed) dataframe. Usually this is a simple exact match, but
    encoding_node can rename a target column if it didn't cleanly convert to
    boolean earlier (e.g. one-hot encoding "churn" with unusual text values
    into "churn_active"/"churn_cancelled" instead of leaving one "churn"
    column behind). This looks for that case too -- any column starting with
    "{target_col}_" -- so a renamed target still gets found instead of
    silently falling back to no-target mode."""
    if not target_col:
        return []
    if target_col in df.columns:
        return [target_col]
    prefix = f"{target_col}_"
    return [c for c in df.columns if c.startswith(prefix)]


def build_insight_context(df: pd.DataFrame, target_col: str = None, engineered_columns: list = None) -> str:
    """Builds a statistical picture of the CLEANED dataframe for the Insight
    Agent. This is deliberately separate from build_profile() (which the
    Planner uses on the raw data) because the Insight Agent needs to reason
    about actual patterns, not just dtypes and nulls.

    Two distinct branches, matching how a target column changes what's
    actually worth showing:

    - TARGET BRANCH (a target column was selected and survived preprocessing):
      compute that column's correlation against every other numeric column --
      including engineered ones, since e.g. "this engineered ratio predicts
      churn" is a genuinely useful thing to know when there IS a specific
      outcome being explained.

    - GENERAL PROFILING BRANCH (no target): deliberately restrict summary
      stats and correlations to ORIGINAL columns only, excluding anything
      feature_engineering_node generated. Without a target to focus the
      search, correlating dozens of automatically engineered ratio/bin/
      cyclical columns against EACH OTHER is exactly the multiple-comparisons
      trap that produced spurious "insights" (e.g. a coincidental 0.87
      correlation between two unrelated engineered columns) in earlier runs.
      Excluding them here means that noise is never even offered to the LLM
      as a candidate insight, rather than relying on the LLM to correctly
      dismiss it."""
    engineered_columns = set(engineered_columns or [])
    parts = [f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns"]

    numeric_df = df.select_dtypes(include=["number"])
    target_cols_found = _find_target_columns(df, target_col)

    if target_col and not target_cols_found:
        parts.append(
            f"NOTE: a target column '{target_col}' was selected, but no matching column "
            f"survived preprocessing (it may have been dropped). Falling back to general "
            f"profiling -- do not assume a target exists."
        )

    if target_cols_found:
        # --- Target branch ---
        parts.append(f"Target column(s): {', '.join(target_cols_found)}")
        parts.append("\nNumeric column statistics (all columns, including engineered features):\n" + numeric_df.describe().to_string())

        target_lines = []
        for tcol in target_cols_found:
            if tcol not in numeric_df.columns or numeric_df.shape[1] < 2:
                continue
            corrs = numeric_df.corr()[tcol].drop(labels=[tcol], errors="ignore").abs().sort_values(ascending=False)
            top = corrs.head(10)
            if not top.empty:
                lines = "\n".join(f"  {col}: {v:.2f}" for col, v in top.items())
                target_lines.append(f"Columns most correlated with '{tcol}' (by |r|):\n{lines}")
        if target_lines:
            parts.append("\n=== TARGET RELATIONSHIPS ===\n" + "\n\n".join(target_lines))

    else:
        # --- General profiling branch: original columns only ---
        original_numeric_df = numeric_df[[c for c in numeric_df.columns if c not in engineered_columns]]

        if not original_numeric_df.empty:
            parts.append(
                "\nNumeric column statistics (original columns only -- engineered "
                "features excluded, see note below):\n" + original_numeric_df.describe().to_string()
            )

            if original_numeric_df.shape[1] > 1:
                corr = original_numeric_df.corr().abs()
                cols = corr.columns
                pairs = [
                    (cols[i], cols[j], corr.iloc[i, j])
                    for i in range(len(cols))
                    for j in range(i + 1, len(cols))
                ]
                pairs.sort(key=lambda p: p[2], reverse=True)
                top_pairs = pairs[:10]
                if top_pairs:
                    pairs_str = "\n".join(f"  {a} <-> {b}: {v:.2f}" for a, b, v in top_pairs)
                    parts.append(f"\nTop correlations among original columns (by |r|):\n{pairs_str}")

        excluded_count = len(engineered_columns & set(df.columns))
        if excluded_count:
            parts.append(
                f"\n(Note: {excluded_count} automatically engineered column(s) were excluded from "
                f"this summary. With no target column to focus the analysis, correlating many "
                f"auto-generated columns against each other tends to surface coincidental patterns "
                f"rather than real ones.)"
            )

    # Categorical columns: excluded from engineered filtering isn't needed for
    # the target branch (kept as-is there), but for general profiling we
    # exclude engineered categoricals too (e.g. feature_engineering_node's
    # "_binned" columns, which are pandas Categorical dtype, not numeric).
    cat_df = df.select_dtypes(include=["object", "category", "bool"])
    if not target_cols_found:
        cat_df = cat_df[[c for c in cat_df.columns if c not in engineered_columns]]
    if not cat_df.empty:
        # Cap at 15 columns so a very wide dataset doesn't blow up the prompt.
        cat_lines = []
        for col in cat_df.columns[:15]:
            top_vals = df[col].value_counts().head(5).to_dict()
            cat_lines.append(f"  {col}: {top_vals}")
        parts.append("\nTop values per categorical column:\n" + "\n".join(cat_lines))

    return "\n".join(parts)


def insight_agent_node(state: GraphState):
    """Second LLM call in the graph. Runs AFTER preprocessing is done, on the
    cleaned dataframe. Decides what's actually worth reporting -- not every
    statistic, just the ones that matter -- and returns them as structured
    Insight objects (converted to plain dicts for state portability)."""
    print("-> Insight Agent: Analyzing cleaned data for noteworthy patterns...")

    df = state["df"]
    target_col = state.get("target_col")
    engineered_columns = state.get("engineered_columns", [])
    context = build_insight_context(df, target_col, engineered_columns)

    llm = get_llm(model=AGENT_MODELS["insight"])
    structured_llm = llm.with_structured_output(InsightReport)

    prompt = f"""You are a senior data analyst reviewing a cleaned dataset.

Identify the most important, specific, and actionable patterns in this data.
Reference real column names and real numbers from the context below -- do not
invent statistics that aren't derivable from it. Skip anything generic or
already obvious (e.g. "the data has {df.shape[0]} rows" is not an insight).

If the context below contains a "TARGET RELATIONSHIPS" section, that is the
main thing you should be explaining -- prioritize insights about what drives
the target column over everything else.

Otherwise, the context reflects general profiling of the dataset's ORIGINAL
columns (automatically engineered columns have already been excluded, so you
don't need to second-guess whether a correlation is a coincidental artifact
of feature engineering -- it isn't).

Context:
{context}
"""

    report = invoke_with_retry(structured_llm, prompt)

    # Sort strongest-first and store as plain dicts (Pydantic objects don't
    # need to survive in LangGraph state -- dicts are simpler to serialize
    # and to hand to the Synthesis Agent later).
    insights = sorted(
        [i.model_dump() for i in report.insights],
        key=lambda i: i["importance"],
        reverse=True,
    )

    print(f"   [+] Found {len(insights)} insights:")
    for ins in insights:
        print(f"      - ({ins['importance']}/5) {ins['title']}")

    return {"insights": insights}


def synthesis_agent_node(state: GraphState):
    """Third LLM call in the graph. Runs AFTER the Insight Agent. Takes the
    structured insights it found and the record of what preprocessing was
    done, and writes them up as one polished Markdown report.

    Deliberately NOT using with_structured_output here (unlike the Planner
    and Insight agents): the output we want is flowing prose/Markdown, and
    forcing that into fixed schema fields would just mean reassembling
    Markdown out of pieces afterwards -- more awkward, no real benefit."""
    print("-> Synthesis Agent: Compiling final report...")

    df = state["df"]
    insights = state.get("insights", [])
    steps_taken = state.get("steps_taken", [])
    charts = state.get("charts", [])
    critic_feedback = state.get("critic_feedback")

    llm = get_llm(model=AGENT_MODELS["synthesis"])

    insights_block = "\n".join(
        f"- [{ins['importance']}/5] {ins['title']}\n"
        f"  {ins['description']}\n"
        f"  Evidence: {ins['supporting_stat']}"
        for ins in insights
    ) or "No insights were found."

    steps_block = ", ".join(steps_taken) if steps_taken else "No preprocessing was needed -- data was already clean."

    charts_block = "\n".join(f"- {c['title']} ({c['path']}): {c['rationale']}" for c in charts) or "No charts were generated."

    # On a revision (the Critic sent this back), tell the LLM exactly what to fix.
    # On a first draft, this is just empty.
    revision_block = (
        f"\n\nIMPORTANT -- a previous draft of this report was reviewed and REJECTED. "
        f"You must specifically address this feedback in your rewrite: {critic_feedback}"
        if critic_feedback else ""
    )

    prompt = f"""You are writing the final report for a data analysis pipeline that
just ran automatically. Write it for a stakeholder who is NOT a data scientist.

Preprocessing steps applied: {steps_block}
Final dataset shape after preprocessing: {df.shape[0]} rows, {df.shape[1]} columns

Insights found (ranked by importance, most important first):
{insights_block}

Charts generated (reference these by title where relevant in Key Findings):
{charts_block}

Write a well-organized Markdown report with these sections:
1. ## Overview -- one paragraph, plain language, what this dataset is and what was done to it.
2. ## Data Preparation Summary -- briefly explain the preprocessing steps applied and why (in plain terms).
3. ## Key Findings -- expand each insight into a short readable paragraph, most important first. Keep the real numbers from the evidence. Mention the relevant chart by title where one exists.
4. ## Recommended Next Steps -- 2-4 concrete, sensible actions based on the findings.

Do not invent any numbers, columns, or findings that aren't given above.{revision_block}
"""

    response = invoke_with_retry(llm, prompt)
    report_markdown = response.content

    print("   [+] Report generated.")

    return {"report_markdown": report_markdown}


class ChartExecutionTimeout(Exception):
    """Raised when generated chart code runs too long -- catches infinite
    loops or accidentally-huge plotting operations."""


def _timeout_handler(signum, frame):
    raise ChartExecutionTimeout("Chart code took too long to run (possible infinite loop)")


# Deliberately small: only what typical matplotlib/seaborn snippets need.
# Notably absent: __import__, open, eval, exec, compile, getattr, globals,
# locals, vars -- so generated code CANNOT import a new module, touch the
# filesystem outside of the one save_path we hand it, or introspect its way
# around the sandbox.
_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in [
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
        "len", "list", "max", "min", "print", "range", "round", "set",
        "sorted", "str", "sum", "tuple", "zip", "isinstance",
    ]
}
_SAFE_BUILTINS.update({"True": True, "False": False, "None": None})


def execute_chart_code(code: str, df: pd.DataFrame, save_path: str, timeout_seconds: int = 15):
    """Runs one chart's LLM-generated code in a restricted exec() sandbox.

    Only df/plt/sns/pd/np/save_path are reachable -- no import statement can
    succeed (no __import__ in the builtins), and no other file on disk can be
    touched (nothing hands the code a real `open`). A SIGALRM-based timeout
    guards against runaway loops. Any exception -- SyntaxError, NameError,
    KeyError from a wrong column name, the timeout, whatever -- propagates up
    to the caller, which is what drives the self-healing retry."""

    sandbox_globals = {
        "__builtins__": _SAFE_BUILTINS,
        "df": df,
        "plt": plt,
        "sns": sns,
        "pd": pd,
        "np": np,
        "save_path": save_path,
    }

    use_alarm = hasattr(signal, "SIGALRM")  # SIGALRM doesn't exist on Windows
    if use_alarm:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)

    try:
        exec(code, sandbox_globals)
    finally:
        if use_alarm:
            signal.alarm(0)
        plt.close("all")  # always clean up figures, success or failure


def _strip_code_fences(text: str) -> str:
    """LLMs love wrapping code in ```python ... ``` even when told not to.
    Strips that off so exec() gets plain Python."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def visualization_agent_node(state: GraphState):
    """Fourth LLM call in the graph, and the only node in the whole pipeline
    where LLM-written code actually executes. Runs after the Insight Agent:
    decides which charts best show off the findings, writes the matplotlib/
    seaborn code for each, and runs it in the sandbox above. If a chart's
    code throws, the real error is sent back to the LLM to fix -- up to
    MAX_CHART_FIX_ATTEMPTS times -- before that one chart is skipped
    (skipping never crashes the graph; charts are a bonus, not a requirement
    for the pipeline to finish)."""
    print("-> Visualization Agent: Planning and generating charts...")

    # .copy() is deliberate: execute_chart_code() hands this exact object to
    # LLM-generated code as `df`. If a chart's code adds a scratch/helper
    # column to make plotting easier (e.g. `df['age_Q4'] = ...`), that's a
    # column ASSIGNMENT, not a reassignment -- it mutates the dataframe
    # object in place. Without this copy, that mutation lands directly on
    # state["df"] (same object), so a chart's private plotting helper column
    # silently leaks into the "final cleaned dataset" everyone downstream
    # (Synthesis Agent, the Streamlit preview/download) sees -- even though
    # it was never part of the actual preprocessing pipeline.
    df = state["df"].copy()
    insights = state.get("insights", [])

    os.makedirs(CHARTS_DIR, exist_ok=True)

    llm = get_llm(model=AGENT_MODELS["visualization"])
    structured_llm = llm.with_structured_output(VisualizationPlan)

    insights_block = "\n".join(
        f"- {ins['title']}: {ins['description']} (Evidence: {ins['supporting_stat']})"
        for ins in insights
    ) or "No specific insights were provided -- use your judgement on what's worth showing."

    columns_block = ", ".join(f"{col} ({dtype})" for col, dtype in df.dtypes.items())

    prompt = f"""You are a data visualization expert. Propose 2 to 5 charts that best
illustrate the findings below, using matplotlib/seaborn.

Available columns and dtypes: {columns_block}

Findings to visualize:
{insights_block}

For each chart, write short, correct Python code using ONLY the pre-provided variables
`df`, `plt`, `sns`, `pd`, `np`, and `save_path` -- do NOT write any import statement,
do NOT use open/exec/eval/os/sys, do NOT reference any file path other than save_path.
End every chart's code with exactly:
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
"""

    plan = invoke_with_retry(structured_llm, prompt)

    generated_charts = []

    for chart in plan.charts:
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", chart.filename) or "chart.png"
        if not safe_name.lower().endswith(".png"):
            safe_name += ".png"
        save_path = os.path.join(CHARTS_DIR, safe_name)

        code = _strip_code_fences(chart.code)
        last_error = None

        for attempt in range(MAX_CHART_FIX_ATTEMPTS + 1):
            try:
                execute_chart_code(code, df, save_path)
                print(f"   [+] Saved chart: {save_path} ({chart.title})")
                generated_charts.append(
                    {"title": chart.title, "rationale": chart.rationale, "path": save_path}
                )
                last_error = None
                break
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                print(f"   [!] Chart '{chart.title}' failed on attempt {attempt + 1}: {last_error}")

                if attempt < MAX_CHART_FIX_ATTEMPTS:
                    # Self-healing loop, scoped to just this one chart's code --
                    # the real traceback goes back to the LLM to fix.
                    fix_llm = get_llm(model=AGENT_MODELS["visualization"])
                    fix_prompt = f"""This matplotlib/seaborn code failed with this error:
{last_error}

Code:
{code}

Rewrite ONLY the code to fix the error. Same rules as before: only use
df/plt/sns/pd/np/save_path, no imports, must end with plt.tight_layout(),
plt.savefig(save_path), plt.close(). Return ONLY the corrected Python code."""
                    fixed = invoke_with_retry(fix_llm, fix_prompt)
                    code = _strip_code_fences(fixed.content)

        if last_error is not None:
            print(f"   [x] Giving up on chart '{chart.title}' after {MAX_CHART_FIX_ATTEMPTS + 1} attempts.")

    print(f"   [+] {len(generated_charts)}/{len(plan.charts)} charts generated successfully.")

    return {"charts": generated_charts}


def critic_agent_node(state: GraphState):
    """Fifth LLM call in the graph. Runs after Synthesis. Reviews the final
    report against the actual insights it's supposed to be based on, and can
    reject it -- sending the graph back to Synthesis with concrete feedback.
    This is the pipeline's real self-healing/reflection step: instead of
    catching a code traceback, it catches a REASONING failure (a report that
    invents numbers, or is too vague to be useful)."""
    print("-> Critic Agent: Reviewing final report...")

    report = state.get("report_markdown", "")
    insights = state.get("insights", [])
    revisions = state.get("critic_revisions", 0)

    insights_block = "\n".join(
        f"- [{ins['importance']}/5] {ins['title']}: {ins['description']} (Evidence: {ins['supporting_stat']})"
        for ins in insights
    ) or "No insights were provided."

    llm = get_llm(model=AGENT_MODELS["critic"])
    structured_llm = llm.with_structured_output(CriticVerdict)

    prompt = f"""You are a skeptical senior reviewer fact-checking an automatically
generated data report. Approve it ONLY if it is fully supported by the source
insights below, specific (not vague filler), and free of invented numbers or claims.

Source insights (the ONLY ground truth -- anything in the report not traceable
to these should be rejected):
{insights_block}

Report to review:
{report}
"""

    verdict = invoke_with_retry(structured_llm, prompt)

    status = "APPROVED" if verdict.approved else "NEEDS REVISION"
    print(f"   [{status}] {verdict.feedback}")

    return {
        "critic_approved": verdict.approved,
        "critic_feedback": verdict.feedback,
        "critic_revisions": revisions + 1,
    }


def route_next(state: GraphState):
    """Conditional-edge function used after the planner and every
    preprocessing step. Reads the remaining plan and returns the name of the
    next node to run. Once the plan is empty, instead of ending the graph we
    now route into the Insight Agent -- preprocessing finishing doesn't mean
    the pipeline is done, just that the data is ready to be analyzed."""
    plan = state["plan"]
    if not plan.steps:
        return "insight_agent"
    return plan.steps[0]


def route_after_critic(state: GraphState):
    """Conditional-edge function after the Critic Agent. Loops back to
    Synthesis for a rewrite if the Critic rejected the draft -- unless we've
    already hit MAX_CRITIC_REVISIONS, in which case we ship the current draft
    anyway rather than risk looping forever on a report the Critic keeps
    disliking for marginal reasons."""
    if state.get("critic_approved"):
        return END
    if state.get("critic_revisions", 0) >= MAX_CRITIC_REVISIONS:
        print(f"   [!] Hit max revisions ({MAX_CRITIC_REVISIONS}) -- shipping current draft as-is.")
        return END
    return "synthesis_agent"


def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("planner", planner_node)
    builder.add_node("data_cleaning", data_cleaning_node)
    builder.add_node("type_conversion", type_conversion_node)
    builder.add_node("imputation", imputation_node)
    builder.add_node("outlier_handling", outlier_handling_node)
    builder.add_node("feature_engineering", feature_engineering_node)
    builder.add_node("encoding", encoding_node)
    builder.add_node("feature_transformation", feature_transformation_node)
    builder.add_node("scaling", scaling_node)
    builder.add_node("dimensionality_reduction", dimensionality_reduction_node)
    builder.add_node("feature_selection", feature_selection_node)
    builder.add_node("insight_agent", insight_agent_node)
    builder.add_node("visualization_agent", visualization_agent_node)
    builder.add_node("synthesis_agent", synthesis_agent_node)
    builder.add_node("critic_agent", critic_agent_node)

    builder.set_entry_point("planner")

    # Every node -- including the planner -- routes to whatever step is next
    # in the plan, or to the Insight Agent once the plan is empty.
    route_map = {step: step for step in VALID_STEPS}
    route_map["insight_agent"] = "insight_agent"

    for node_name in ["planner"] + VALID_STEPS:
        builder.add_conditional_edges(node_name, route_next, route_map)

    # Full multi-agent chain:
    #   Insight -> Visualization -> Synthesis -> Critic
    #                                    ^            |
    #                                    └── reject ───┘ (up to MAX_CRITIC_REVISIONS times)
    builder.add_edge("insight_agent", "visualization_agent")
    builder.add_edge("visualization_agent", "synthesis_agent")
    builder.add_edge("synthesis_agent", "critic_agent")
    builder.add_conditional_edges(
        "critic_agent",
        route_after_critic,
        {"synthesis_agent": "synthesis_agent", END: END},
    )

    return builder.compile()


if __name__ == "__main__":
    # Quick manual smoke test:
    #   python -m src.graph path/to/your.csv [target_column]
    # target_column is optional -- the raw column name as it appears in the
    # CSV (e.g. "Churn", not "churn"). If omitted, the Planner will try to
    # guess a target on its own; if it can't, the pipeline runs in general
    # profiling mode.
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.graph <path_to_csv> [target_column]")
        sys.exit(1)

    graph = build_graph()
    initial_state: GraphState = {"df": pd.read_csv(sys.argv[1])}
    if len(sys.argv) >= 3:
        initial_state["target_col"] = sys.argv[2]
    final_state = graph.invoke(initial_state)

    print("\n=== Final DataFrame ===")
    print(final_state["df"].head())
    print(f"\nFinal shape: {final_state['df'].shape}")

    print("\n=== Insights ===")
    for ins in final_state.get("insights", []):
        print(f"\n[{ins['importance']}/5] {ins['title']}")
        print(f"  {ins['description']}")
        print(f"  Evidence: {ins['supporting_stat']}")

    print("\n=== Charts ===")
    charts = final_state.get("charts", [])
    if charts:
        for c in charts:
            print(f"  {c['path']}: {c['title']}")
    else:
        print("  (none generated)")

    print(f"\n=== Critic ===")
    print(f"  Approved: {final_state.get('critic_approved')}")
    print(f"  Revisions used: {final_state.get('critic_revisions', 0)}")
    print(f"  Final feedback: {final_state.get('critic_feedback')}")

    print("\n=== Final Report ===")
    report = final_state.get("report_markdown", "")
    print(report)

    with open("report.md", "w") as f:
        f.write(report)
    print("\n[+] Report also saved to report.md")
    print(f"[+] Charts saved under ./{CHARTS_DIR}/")