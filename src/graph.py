import ast
import builtins
import io
import os
import re
import signal
import time
from pathlib import Path
from groq import RateLimitError, BadRequestError
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

AGENT_MODELS = {
    "planner": "openai/gpt-oss-20b",
    "insight": "openai/gpt-oss-120b",
    "synthesis": "openai/gpt-oss-20b",
    "visualization": "openai/gpt-oss-120b",
    "critic": "openai/gpt-oss-120b",
}


CHARTS_DIR = os.getenv(
    "CHARTS_DIR",
    str(Path(__file__).resolve().parent.parent / "charts"),
)
MAX_CHART_FIX_ATTEMPTS = 2  # how many times we let the LLM try to fix its own broken chart code
MAX_CRITIC_REVISIONS = 2  # how many times the Critic can send the report back before we ship it anyway


RATE_LIMIT_MAX_ATTEMPTS = 4
RATE_LIMIT_BASE_WAIT_SECONDS = 15  # attempt 1 waits 15s, attempt 2 waits 30s, attempt 3 waits 45s...


TOOL_USE_FAILED_MAX_ATTEMPTS = 3
TOOL_USE_FAILED_WAIT_SECONDS = 3  # short and fixed -- this isn't a rate-limit bucket refilling, just a reroll


def _is_tool_use_failed(e: BadRequestError) -> bool:
    body = getattr(e, "body", None) or {}
    error = body.get("error", {}) if isinstance(body, dict) else {}
    return error.get("code") == "tool_use_failed"


def invoke_with_retry(runnable, prompt):
    """Wraps a .invoke() call (on either a plain LLM or a
    with_structured_output-wrapped one) with a real wait-and-retry loop
    covering two distinct Groq failure modes: RateLimitError (HTTP 429) and
    the "tool_use_failed" flavor of BadRequestError (HTTP 400) described
    above. One flat attempt loop handles both -- each attempt either
    succeeds, hits a retryable error (and sleeps an amount appropriate to
    THAT error type before the next attempt), or hits a non-retryable error
    and raises immediately. Every direct LLM call in this file goes through
    this instead of calling .invoke() directly."""
    max_attempts = max(RATE_LIMIT_MAX_ATTEMPTS, TOOL_USE_FAILED_MAX_ATTEMPTS)
    last_error: Exception = RuntimeError("invoke_with_retry failed")
    for attempt in range(max_attempts):
        try:
            return runnable.invoke(prompt)
        except RateLimitError as e:
            last_error = e
            if attempt >= RATE_LIMIT_MAX_ATTEMPTS - 1:
                break
            wait = RATE_LIMIT_BASE_WAIT_SECONDS * (attempt + 1)
            print(f"   [!] Rate limited -- waiting {wait}s before retry ({attempt + 1}/{RATE_LIMIT_MAX_ATTEMPTS})...")
            time.sleep(wait)
        except BadRequestError as e:
            if not _is_tool_use_failed(e):
                raise  # a real malformed request -- don't mask it with a retry
            last_error = e
            if attempt >= TOOL_USE_FAILED_MAX_ATTEMPTS - 1:
                break
            print(
                f"   [!] Model didn't call the required tool (tool_use_failed) -- "
                f"waiting {TOOL_USE_FAILED_WAIT_SECONDS}s before retry "
                f"({attempt + 1}/{TOOL_USE_FAILED_MAX_ATTEMPTS})..."
            )
            time.sleep(TOOL_USE_FAILED_WAIT_SECONDS)
    raise last_error



MAX_DESCRIBE_COLS = int(os.getenv("MAX_DESCRIBE_COLS", 25))
MAX_COLUMNS_LISTED = int(os.getenv("MAX_COLUMNS_LISTED", 40))

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

        target_lines = []
        top_corr_cols = []
        for tcol in target_cols_found:
            if tcol not in numeric_df.columns or numeric_df.shape[1] < 2:
                continue
            corrs = numeric_df.corr()[tcol].drop(labels=[tcol], errors="ignore").abs().sort_values(ascending=False)
            top = corrs.head(10)
            top_corr_cols.extend(top.index.tolist())
            if not top.empty:
                lines = "\n".join(f"  {col}: {v:.2f}" for col, v in top.items())
                target_lines.append(f"Columns most correlated with '{tcol}' (by |r|):\n{lines}")

        describe_cols = list(dict.fromkeys(list(target_cols_found) + top_corr_cols))[:MAX_DESCRIBE_COLS]
        stats_df = numeric_df[[c for c in describe_cols if c in numeric_df.columns]]
        if not stats_df.empty:
            omitted = numeric_df.shape[1] - stats_df.shape[1]
            note = f" -- {omitted} other numeric column(s) omitted for brevity" if omitted > 0 else ""
            parts.append(
                f"\nNumeric column statistics (target + top correlated columns only{note}):\n"
                + stats_df.describe().to_string()
            )

        if target_lines:
            parts.append("\n=== TARGET RELATIONSHIPS ===\n" + "\n\n".join(target_lines))

    else:
        # --- General profiling branch: original columns only ---
        original_numeric_df = numeric_df[[c for c in numeric_df.columns if c not in engineered_columns]]

        if not original_numeric_df.empty:
            
            describe_df = original_numeric_df
            omitted_note = ""
            if describe_df.shape[1] > MAX_DESCRIBE_COLS:
                top_var_cols = describe_df.var().nlargest(MAX_DESCRIBE_COLS).index.tolist()
                omitted = describe_df.shape[1] - MAX_DESCRIBE_COLS
                describe_df = describe_df[top_var_cols]
                omitted_note = f" -- {omitted} other original numeric column(s) omitted for brevity"
            parts.append(
                "\nNumeric column statistics (original columns only -- engineered "
                f"features excluded{omitted_note}):\n" + describe_df.describe().to_string()
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



_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in [
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
        "len", "list", "max", "min", "print", "range", "round", "set",
        "sorted", "str", "sum", "tuple", "zip", "isinstance",
    ]
}
_SAFE_BUILTINS.update({"True": True, "False": False, "None": None})

# Module prefixes the plotting stack is allowed to lazily import at runtime.
_IMPORT_ALLOWLIST = ("numpy", "pandas", "matplotlib", "seaborn", "scipy", "mpl_toolkits")


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    """A deliberately narrow __import__ for the sandbox.

    This exists because leaving __import__ out entirely BREAKS legitimate
    chart code. numpy defers part of itself: the first `arr.mean()` call
    triggers an import of `numpy._core._methods`, and that import resolves
    through the CALLING frame's __builtins__ -- i.e. ours. With no __import__
    there, a plain `np.array([...]).mean()` dies with KeyError: '__import__'.
    (That was a live bug in this sandbox before these tests existed; charts
    using .mean()/.std() failed and got silently retried or dropped.)

    So instead of removing it, we constrain it: only modules under the
    plotting stack's own namespaces resolve. os, sys, subprocess, importlib,
    builtins and everything else raise ImportError. Combined with the AST
    guard -- which rejects `import` statements and the name `__import__`
    outright -- user code cannot reach this at all; only library internals
    can, which is exactly who needs it."""
    root = name.split(".")[0]
    if root not in _IMPORT_ALLOWLIST:
        raise ImportError(f"import of '{name}' is not permitted in chart code")
    return builtins.__import__(name, globals, locals, fromlist, level)


_SAFE_BUILTINS["__import__"] = _guarded_import


class UnsafeChartCode(Exception):
    """Raised when generated chart code fails the static safety check, before
    it is ever executed. Carries a plain-English reason, which gets fed back
    to the LLM by the existing self-healing retry loop."""



_FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.Lambda,          # a lambda body sidesteps nothing, but nothing needs one either
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
)


_FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "input", "breakpoint",
    "getattr", "setattr", "delattr", "hasattr",
    "globals", "locals", "vars", "dir",
    "memoryview", "__import__",
}


def _assert_chart_code_is_safe(code: str):
    """Static gate that runs BEFORE exec(). Parses the code and rejects
    anything that could reach outside the plotting sandbox.

    This is the real boundary. It's an allowlist-shaped check on a small,
    well-understood surface (a plotting snippet), which is exactly the case
    where static analysis is tractable -- we know what legitimate chart code
    looks like, and none of it needs dunder attributes or introspection
    builtins.

    Raises UnsafeChartCode with a specific reason, which the caller feeds
    back to the LLM as a fix-it message."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeChartCode(f"code does not parse: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise UnsafeChartCode(
                f"'{type(node).__name__}' is not allowed in chart code -- write a "
                f"plain sequence of plotting statements using only df/plt/sns/pd/np/save_path."
            )

        # Any dunder, in any position, is an escape vector. Reject on sight.
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeChartCode(
                f"attribute '{node.attr}' is not allowed (double-underscore attributes "
                f"can be used to break out of the sandbox)."
            )
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise UnsafeChartCode(f"name '{node.id}' is not allowed.")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise UnsafeChartCode(f"'{node.id}' is not allowed in chart code.")
        # A dunder hidden in a string literal is only useful with getattr,
        # which is already blocked -- but reject it anyway so the intent is
        # never ambiguous when reviewing logs.
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("__") and node.value.endswith("__"):
                raise UnsafeChartCode(
                    f"string literal '{node.value}' looks like a dunder lookup and is not allowed."
                )


def _make_write_jail(charts_dir: str):
    """Returns a plt proxy whose savefig() can only write inside charts_dir.

    Blocking imports doesn't stop `plt.savefig('/etc/anything')` -- matplotlib
    is a legitimately-provided object that already holds filesystem access, so
    the write-anywhere hole survives every builtins restriction. This wraps the
    one method that writes, resolves the requested path, and refuses anything
    that lands outside the charts directory (which also covers '..' traversal,
    because resolve() normalizes it first)."""
    charts_root = Path(charts_dir).resolve()

    class _JailedPlt:
        def __getattr__(self, name):
            # Everything except savefig passes straight through to matplotlib.
            return getattr(plt, name)

        def savefig(self, fname, *args, **kwargs):
            try:
                target = Path(str(fname)).resolve()
            except Exception as e:
                raise UnsafeChartCode(f"invalid save path: {fname!r}") from e
            if not target.is_relative_to(charts_root):
                raise UnsafeChartCode(
                    f"chart code tried to write to {target} -- writes are only "
                    f"permitted inside {charts_root}. Use the provided `save_path`."
                )
            return plt.savefig(target, *args, **kwargs)

    return _JailedPlt()


def execute_chart_code(code: str, df: pd.DataFrame, save_path: str, timeout_seconds: int = 15):
    """Runs one chart's LLM-generated code in a restricted sandbox.

    Three layers, in order:
      1. A static AST check (_assert_chart_code_is_safe) that rejects imports,
         dunder access and introspection builtins BEFORE anything executes.
         This is the layer that actually prevents sandbox escape.
      2. A minimal __builtins__ allowlist, so even if something slipped past
         layer 1 there is no open/eval to call and imports are restricted to
         the plotting stack.
      3. A write jail on plt.savefig, so generated code cannot write outside
         the charts directory even though matplotlib itself can.
    A SIGALRM timeout guards against runaway loops.

    Any exception -- UnsafeChartCode, SyntaxError, a KeyError from a wrong
    column name, the timeout -- propagates to the caller, which is what drives
    the self-healing retry.

    A note on scope: this is defense against a confused or prompt-injected
    LLM writing dangerous code, which is the realistic threat here (column
    names from an uploaded CSV flow into the prompt that generates this code).
    It is NOT a substitute for OS-level isolation if you ever run genuinely
    untrusted code -- for that you want a container or gVisor, not an
    in-process check."""

    _assert_chart_code_is_safe(code)

    # Constrain writes to the directory the chart is meant to go in.
    charts_dir = os.path.dirname(os.path.abspath(save_path)) or CHARTS_DIR

    sandbox_globals = {
        "__builtins__": _SAFE_BUILTINS,
        "df": df,
        "plt": _make_write_jail(charts_dir),
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

    
    df = state["df"].copy()
    insights = state.get("insights", [])

    os.makedirs(CHARTS_DIR, exist_ok=True)

    llm = get_llm(model=AGENT_MODELS["visualization"])
    structured_llm = llm.with_structured_output(VisualizationPlan)

    insights_block = "\n".join(
        f"- {ins['title']}: {ins['description']} (Evidence: {ins['supporting_stat']})"
        for ins in insights
    ) or "No specific insights were provided -- use your judgement on what's worth showing."

    
    all_columns = list(df.dtypes.items())
    mentioned = [(col, dtype) for col, dtype in all_columns if str(col) in insights_block]
    mentioned_names = {col for col, _ in mentioned}
    remaining = [(col, dtype) for col, dtype in all_columns if col not in mentioned_names]
    shown_columns = (mentioned + remaining)[:MAX_COLUMNS_LISTED]
    omitted_count = len(all_columns) - len(shown_columns)

    columns_block = ", ".join(f"{col} ({dtype})" for col, dtype in shown_columns)
    if omitted_count > 0:
        columns_block += f", ... and {omitted_count} more column(s) not shown"

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
            except UnsafeChartCode as e:
                # A safety rejection is louder than an ordinary bug: it means
                # the model emitted code that tried to leave the sandbox. That
                # can be a confused generation, but it can also be the tail of
                # a prompt injection carried in a column name from an uploaded
                # CSV -- so it's logged distinctly rather than blending into
                # the normal failure stream.
                last_error = f"SAFETY REJECTION: {e}"
                print(f"   [!!] Chart '{chart.title}' REJECTED by the safety guard: {e}")

                if attempt < MAX_CHART_FIX_ATTEMPTS:
                    fix_llm = get_llm(model=AGENT_MODELS["visualization"])
                    fix_prompt = f"""The code below was REJECTED by a safety check before running.

Reason: {e}

Code:
{code}

Rewrite it as a plain sequence of plotting statements. You may ONLY use the
pre-provided variables df, plt, sns, pd, np and save_path. No imports, no
function or class definitions, no lambdas, no attribute names starting with
underscores, and no eval/exec/open/getattr. It must end with
plt.tight_layout(), plt.savefig(save_path), plt.close().
Return ONLY the corrected Python code."""
                    fixed = invoke_with_retry(fix_llm, fix_prompt)
                    code = _strip_code_fences(fixed.content)
                continue
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