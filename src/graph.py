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


AGENT_MODELS = {
    "planner": None,
    "insight": "openai/gpt-oss-120b",
}

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

    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    null_counts=df.isnull().sum()
    null_summary = null_counts[null_counts > 0]

    info_str += f"\n\nNull values:\n{null_summary.to_string() if len(null_summary) > 0 else 'None'}"

    info_str += "\n\nSample Data (First 5 rows): \n"
    info_str += df.head().to_string()

    return info_str

def planner_node(state: GraphState):
    df=state["df"]
    profile = build_profile(df)

    llm=get_llm()
    structured_llm = llm.with_structured_output(PreprocessingPlan)

    prompt =f"""You are a senior data scientist planning an EDA/preprocessing pipeline for the dataset profiled below.
    Choose the ordered sequence of preprocessing steps this specific dataset needs.
    Do Not include a step just because it exists -- only include steps that this dataset's profile actually justifies (e.g. skip 'imputation' if there are no nulls, skip 'dimensionality_reduction' if there aren't many columns).
    Valid step names (use these exact strings, nothing else): {",".join(VALID_STEPS)}
    
    Dataset profile:
    {profile}
    
    
    
"""


    plan = structured_llm.invoke(prompt)


    plan.steps = [s for s in plan.steps if s in VALID_STEPS]

    print(f" [+]Plan: {plan.steps}")
    print(f" [+]Reasoning: {plan.reasoning}")


