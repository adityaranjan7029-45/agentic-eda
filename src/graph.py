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

valid_steps = [
    "data_cleaning",
]