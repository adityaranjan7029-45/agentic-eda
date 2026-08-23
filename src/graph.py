import io
import pandas as pd

from langgraph.graph import StateGraph, END

from src.config import get_llm
from src.nodes import (
    GraphState,
    PreprocessingPlan,
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
# pyrefly: ignore [missing-import]
from src.schemas import InsightReport

# Which Groq model each agent uses. Kept as one dict so model choices are
# visible/tunable in one place instead of scattered through node functions.
AGENT_MODELS = {
    "planner": None,  # None = fall back to GROQ_MODEL in .env (default llama-3.3-70b-versatile)
    "insight": "openai/gpt-oss-120b",
    "synthesis": "llama-3.1-8b-instant",  # fast/cheap model -- this is a writing/formatting task, not deep reasoning
}

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

    llm = get_llm()
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

    plan = structured_llm.invoke(prompt)

    # Safety net: silently drop any step name the LLM invents that isn't in
    # our fixed vocabulary, so a hallucinated step can't crash the router.
    plan.steps = [s for s in plan.steps if s in VALID_STEPS]

    print(f"   [+] Plan: {plan.steps}")
    print(f"   [+] Reasoning: {plan.reasoning}")

    # Freeze a copy of the chosen steps now, before any node starts popping
    # them off plan.steps -- otherwise by the time the Synthesis Agent runs,
    # plan.steps is empty and we've lost the record of what actually happened.
    return {"plan": plan, "steps_taken": list(plan.steps)}


def build_insight_context(df: pd.DataFrame, target_col: str = None) -> str:
    """Builds a richer statistical picture of the CLEANED dataframe for the
    Insight Agent -- summary stats, top correlations, and top category values.
    This is deliberately separate from build_profile() (which the Planner
    uses on the raw data) because the Insight Agent needs to reason about
    actual patterns, not just dtypes and nulls."""
    parts = [f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns"]

    if target_col and target_col in df.columns:
        parts.append(f"Target column: {target_col}")

    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        parts.append("\nNumeric column statistics:\n" + numeric_df.describe().to_string())

        # Top correlated pairs, so the LLM doesn't have to eyeball a full matrix.
        if numeric_df.shape[1] > 1:
            corr = numeric_df.corr().abs()
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
                parts.append(f"\nTop correlations (by |r|):\n{pairs_str}")

    cat_df = df.select_dtypes(include=["object", "category", "bool"])
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
    context = build_insight_context(df, target_col)

    llm = get_llm(model=AGENT_MODELS["insight"])
    structured_llm = llm.with_structured_output(InsightReport)

    prompt = f"""You are a senior data analyst reviewing a cleaned dataset.

Identify the most important, specific, and actionable patterns in this data.
Reference real column names and real numbers from the context below -- do not
invent statistics that aren't derivable from it. Skip anything generic or
already obvious (e.g. "the data has {df.shape[0]} rows" is not an insight).

Context:
{context}
"""

    report = structured_llm.invoke(prompt)

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

    llm = get_llm(model=AGENT_MODELS["synthesis"])

    insights_block = "\n".join(
        f"- [{ins['importance']}/5] {ins['title']}\n"
        f"  {ins['description']}\n"
        f"  Evidence: {ins['supporting_stat']}"
        for ins in insights
    ) or "No insights were found."

    steps_block = ", ".join(steps_taken) if steps_taken else "No preprocessing was needed -- data was already clean."

    prompt = f"""You are writing the final report for a data analysis pipeline that
just ran automatically. Write it for a stakeholder who is NOT a data scientist.

Preprocessing steps applied: {steps_block}
Final dataset shape after preprocessing: {df.shape[0]} rows, {df.shape[1]} columns

Insights found (ranked by importance, most important first):
{insights_block}

Write a well-organized Markdown report with these sections:
1. ## Overview -- one paragraph, plain language, what this dataset is and what was done to it.
2. ## Data Preparation Summary -- briefly explain the preprocessing steps applied and why (in plain terms).
3. ## Key Findings -- expand each insight into a short readable paragraph, most important first. Keep the real numbers from the evidence.
4. ## Recommended Next Steps -- 2-4 concrete, sensible actions based on the findings.

Do not invent any numbers, columns, or findings that aren't given above.
"""

    response = llm.invoke(prompt)
    report_markdown = response.content

    print("   [+] Report generated.")

    return {"report_markdown": report_markdown}


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
    builder.add_node("synthesis_agent", synthesis_agent_node)

    builder.set_entry_point("planner")

    # Every node -- including the planner -- routes to whatever step is next
    # in the plan, or to the Insight Agent once the plan is empty.
    route_map = {step: step for step in VALID_STEPS}
    route_map["insight_agent"] = "insight_agent"

    for node_name in ["planner"] + VALID_STEPS:
        builder.add_conditional_edges(node_name, route_next, route_map)

    # Insight Agent always flows into Synthesis. Synthesis is the last node
    # for now -- a Visualization Agent (and optionally a Critic Agent) can be
    # inserted into this chain later without touching anything before it.
    builder.add_edge("insight_agent", "synthesis_agent")
    builder.add_edge("synthesis_agent", END)

    return builder.compile()


if __name__ == "__main__":
    # Quick manual smoke test:
    #   python -m src.graph path/to/your.csv
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.graph <path_to_csv>")
        sys.exit(1)

    graph = build_graph()
    initial_state: GraphState = {"df": pd.read_csv(sys.argv[1])}
    final_state = graph.invoke(initial_state)

    print("\n=== Final DataFrame ===")
    print(final_state["df"].head())
    print(f"\nFinal shape: {final_state['df'].shape}")

    print("\n=== Insights ===")
    for ins in final_state.get("insights", []):
        print(f"\n[{ins['importance']}/5] {ins['title']}")
        print(f"  {ins['description']}")
        print(f"  Evidence: {ins['supporting_stat']}")

    print("\n=== Final Report ===")
    report = final_state.get("report_markdown", "")
    print(report)

    with open("report.md", "w") as f:
        f.write(report)
    print("\n[+] Report also saved to report.md")
