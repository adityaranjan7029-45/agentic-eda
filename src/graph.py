import ast
import builtins
import io
import os
import re
import sys
import time
import uuid
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
TOOL_USE_FAILED_WAIT_SECONDS = 3  # short and fixed -- just a reroll, not a rate-limit window


def _is_tool_use_failed(e: BadRequestError) -> bool:
    body = getattr(e, "body", None) or {}
    error = body.get("error", {}) if isinstance(body, dict) else {}
    return error.get("code") == "tool_use_failed"


def invoke_with_retry(runnable, prompt):
    """Retries a .invoke() call through both RateLimitError and tool_use_failed. Every direct LLM call in this file goes through this."""
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
    """Same style of profile as profile_dataframe(), but reads the graph state's df directly."""
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
    """The one planning LLM call: decides which of VALID_STEPS apply, and in what order."""
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

    # Silently drop any hallucinated step name outside our fixed vocabulary.
    plan.steps = [s for s in plan.steps if s in VALID_STEPS]

    print(f"   [+] Plan: {plan.steps}")
    print(f"   [+] Reasoning: {plan.reasoning}")

    # Target column comes only from state["target_col"] (the Streamlit
    # dropdown), checked against the raw columns then normalized so it still
    # matches after cleaning.
    raw_target = state.get("target_col")
    target_col = normalize_column_name(raw_target) if raw_target and raw_target in df.columns else None

    if target_col:
        print(f"   [+] Target column: '{target_col}'")
    else:
        print("   [+] No target column selected -- running in general profiling mode.")

    # Freeze the chosen steps now, before nodes start popping them off plan.steps.
    return {"plan": plan, "steps_taken": list(plan.steps), "target_col": target_col}


def _find_target_columns(df: pd.DataFrame, target_col: str):
    """Resolves target_col against the actual (preprocessed) columns, including a one-hot-renamed variant like 'churn_active'."""
    if not target_col:
        return []
    if target_col in df.columns:
        return [target_col]
    prefix = f"{target_col}_"
    return [c for c in df.columns if c.startswith(prefix)]


def build_insight_context(df: pd.DataFrame, target_col: str = None, engineered_columns: list = None) -> str:
    """Statistical picture of the cleaned dataframe for the Insight Agent -- correlated-with-target if a target exists, otherwise original columns only (engineered columns excluded to avoid multiple-comparisons noise)."""
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

        # Cap describe() to the target plus its top-correlated columns, not
        # every numeric column, to stay under Groq's token budget.
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
            # Same token-budget cap, prioritizing highest-variance columns.
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

    # Categorical columns: for general profiling, exclude engineered ones too.
    cat_df = df.select_dtypes(include=["object", "category", "bool"])
    if not target_cols_found:
        cat_df = cat_df[[c for c in cat_df.columns if c not in engineered_columns]]
    if not cat_df.empty:
        cat_lines = []
        for col in cat_df.columns[:15]:  # cap at 15 for very wide datasets
            top_vals = df[col].value_counts().head(5).to_dict()
            cat_lines.append(f"  {col}: {top_vals}")
        parts.append("\nTop values per categorical column:\n" + "\n".join(cat_lines))

    return "\n".join(parts)


def insight_agent_node(state: GraphState):
    """Runs on the cleaned dataframe. Decides what's worth reporting and returns Insight objects as plain dicts."""
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
    """Writes the insights and preprocessing record up as one Markdown report. Plain LLM call, not structured output, since the goal is flowing prose."""
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
    """Raised when generated chart code runs too long (possible infinite loop)."""


# Filename the sandboxed code is compiled under, so the timeout tracer can
# tell the chart's own frames apart from matplotlib/pandas internals.
_SANDBOX_FILENAME = "<chart-code>"


def _make_deadline_tracer(deadline: float):
    """sys.settrace hook that aborts the sandboxed code once `deadline` passes.

    Replaces the old SIGALRM timeout, which only works in the main thread.
    Streamlit runs every script in a worker thread, so on Linux SIGALRM raised
    ValueError and every chart failed. settrace is per-thread and works on
    Windows too. Only frames compiled from the chart code are line-traced, so
    matplotlib's own internals run at full speed."""

    def local_tracer(frame, event, arg):
        if time.monotonic() > deadline:
            raise ChartExecutionTimeout("Chart code took too long to run (possible infinite loop)")
        return local_tracer

    def global_tracer(frame, event, arg):
        if frame.f_code.co_filename == _SANDBOX_FILENAME:
            return local_tracer
        return None

    return global_tracer

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
    """Narrow __import__ for the sandbox -- only the plotting stack's own namespaces resolve (numpy needs this for its lazy internal imports); os/sys/subprocess/etc raise ImportError."""
    root = name.split(".")[0]
    if root not in _IMPORT_ALLOWLIST:
        raise ImportError(f"import of '{name}' is not permitted in chart code")
    return builtins.__import__(name, globals, locals, fromlist, level)


_SAFE_BUILTINS["__import__"] = _guarded_import


class UnsafeChartCode(Exception):
    """Raised when generated chart code fails the static safety check, before execution. Fed back to the LLM by the self-healing retry loop."""


# Node types with no legitimate place in a self-contained plotting snippet.
_FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.Lambda,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
)

# Any dunder is rejected wholesale (not individually blocklisted) since the
# escape chains are open-ended (__class__, __bases__, __subclasses__, ...).
_FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "input", "breakpoint",
    "getattr", "setattr", "delattr", "hasattr",
    "globals", "locals", "vars", "dir",
    "memoryview", "__import__",
}


def _assert_chart_code_is_safe(code: str):
    """Static AST gate that runs before exec() -- the real sandbox boundary. Raises UnsafeChartCode with a specific reason, fed back to the LLM as a fix-it message."""
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

        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeChartCode(
                f"attribute '{node.attr}' is not allowed (double-underscore attributes "
                f"can be used to break out of the sandbox)."
            )
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise UnsafeChartCode(f"name '{node.id}' is not allowed.")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise UnsafeChartCode(f"'{node.id}' is not allowed in chart code.")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("__") and node.value.endswith("__"):
                raise UnsafeChartCode(
                    f"string literal '{node.value}' looks like a dunder lookup and is not allowed."
                )


def _make_write_jail(charts_dir: str):
    """Returns a plt proxy whose savefig() can only write inside charts_dir -- matplotlib already has filesystem access, so this is the layer that actually stops path traversal / arbitrary writes."""
    charts_root = Path(charts_dir).resolve()

    class _JailedPlt:
        def __getattr__(self, name):
            return getattr(plt, name)  # everything except savefig passes straight through

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
    """Runs one chart's LLM-generated code in a restricted sandbox: AST safety check, minimal builtins, a savefig write jail, and a per-thread sys.settrace timeout. Any exception propagates to drive the self-healing retry. Defense against a confused/prompt-injected LLM, not OS-level isolation."""

    _assert_chart_code_is_safe(code)

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

    compiled = compile(code, _SANDBOX_FILENAME, "exec")
    previous_tracer = sys.gettrace()
    sys.settrace(_make_deadline_tracer(time.monotonic() + timeout_seconds))
    try:
        exec(compiled, sandbox_globals)
    finally:
        sys.settrace(previous_tracer)
        plt.close("all")  # always clean up figures, success or failure


def _strip_code_fences(text: str) -> str:
    """Strips ```python ... ``` wrapping so exec() gets plain Python."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


# Punctuation matplotlib's default font can't always render (falls back to a
# tofu box, e.g. "Fiber-Optic" -> "Fiber□Optic"). Normalizing to plain ASCII
# is lossless for display purposes.
_UNICODE_PUNCTUATION_MAP = {
    "‐": "-",   # hyphen
    "‑": "-",   # non-breaking hyphen
    "‒": "-",   # figure dash
    "–": "-",   # en dash
    "—": "-",   # em dash
    "―": "-",   # horizontal bar
    "‘": "'",   # left single quote
    "’": "'",   # right single quote
    "“": '"',   # left double quote
    "”": '"',   # right double quote
    "…": "...", # ellipsis
    " ": " ",   # non-breaking space
    "−": "-",   # minus sign
}


def _normalize_unicode_punctuation(text: str) -> str:
    """Replaces punctuation that commonly renders as a missing-glyph box in matplotlib's default font with a plain-ASCII equivalent."""
    if not text:
        return text
    for bad, good in _UNICODE_PUNCTUATION_MAP.items():
        text = text.replace(bad, good)
    return text


def visualization_agent_node(state: GraphState):
    """Decides which charts to draw, writes matplotlib/seaborn code for each, and runs it in the sandbox above. A chart's code that throws gets the real error fed back to the LLM to fix, up to MAX_CHART_FIX_ATTEMPTS, before being skipped."""
    print("-> Visualization Agent: Planning and generating charts...")

    
    df = state["df"].copy()
    insights = state.get("insights", [])

    # One directory per run: charts from concurrent users can't overwrite
    # each other even when the LLM picks the same filename.
    run_charts_dir = os.path.join(CHARTS_DIR, uuid.uuid4().hex[:12])
    os.makedirs(run_charts_dir, exist_ok=True)

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
        save_path = os.path.join(run_charts_dir, safe_name)

        code = _normalize_unicode_punctuation(_strip_code_fences(chart.code))
        chart_title = _normalize_unicode_punctuation(chart.title)
        chart_rationale = _normalize_unicode_punctuation(chart.rationale)
        last_error = None

        for attempt in range(MAX_CHART_FIX_ATTEMPTS + 1):
            try:
                execute_chart_code(code, df, save_path)
                print(f"   [+] Saved chart: {save_path} ({chart_title})")
                generated_charts.append(
                    {"title": chart_title, "rationale": chart_rationale, "path": save_path}
                )
                last_error = None
                break
            except UnsafeChartCode as e:
                # Logged distinctly from ordinary failures: could be a
                # confused generation, or a prompt injection via a CSV column name.
                last_error = f"SAFETY REJECTION: {e}"
                print(f"   [!!] Chart '{chart_title}' REJECTED by the safety guard: {e}")

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
                    code = _normalize_unicode_punctuation(_strip_code_fences(fixed.content))
                continue
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                print(f"   [!] Chart '{chart_title}' failed on attempt {attempt + 1}: {last_error}")

                if attempt < MAX_CHART_FIX_ATTEMPTS:
                    fix_llm = get_llm(model=AGENT_MODELS["visualization"])
                    fix_prompt = f"""This matplotlib/seaborn code failed with this error:
{last_error}

Code:
{code}

Rewrite ONLY the code to fix the error. Same rules as before: only use
df/plt/sns/pd/np/save_path, no imports, must end with plt.tight_layout(),
plt.savefig(save_path), plt.close(). Return ONLY the corrected Python code."""
                    fixed = invoke_with_retry(fix_llm, fix_prompt)
                    code = _normalize_unicode_punctuation(_strip_code_fences(fixed.content))

        if last_error is not None:
            print(f"   [x] Giving up on chart '{chart_title}' after {MAX_CHART_FIX_ATTEMPTS + 1} attempts.")

    print(f"   [+] {len(generated_charts)}/{len(plan.charts)} charts generated successfully.")

    return {"charts": generated_charts}


def critic_agent_node(state: GraphState):
    """Reviews the final report against the actual insights, and can reject it -- sending the graph back to Synthesis with concrete feedback. Catches reasoning failures (invented numbers, vague filler), not code errors."""
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
    """Reads the remaining plan and returns the next node's name; once empty, routes into the Insight Agent."""
    plan = state["plan"]
    if not plan.steps:
        return "insight_agent"
    return plan.steps[0]


def route_after_critic(state: GraphState):
    """Loops back to Synthesis on rejection, unless MAX_CRITIC_REVISIONS is hit -- then ships the current draft anyway."""
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

    route_map = {step: step for step in VALID_STEPS}
    route_map["insight_agent"] = "insight_agent"

    for node_name in ["planner"] + VALID_STEPS:
        builder.add_conditional_edges(node_name, route_next, route_map)

    # Insight -> Visualization -> Synthesis -> Critic, looping back to
    # Synthesis on rejection (up to MAX_CRITIC_REVISIONS times).
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
    # python -m src.graph path/to/your.csv [target_column]
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