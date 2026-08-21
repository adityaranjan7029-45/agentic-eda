"""
Shared Pydantic schemas for the multi-agent pipeline. Kept separate from
src/nodes.py (which already defines PreprocessingPlan) so this file can grow
with the Visualization/Critic/Synthesis schemas later without nodes.py
turning into a dumping ground.
"""

from typing import List
from pydantic import BaseModel, Field


class Insight(BaseModel):
    title: str = Field(
        description="A short, specific headline for this insight -- name the "
        "actual columns/values involved, not a generic label. "
        "e.g. 'Month-to-month contracts churn 4x more than annual ones', "
        "not 'Interesting churn pattern'."
    )
    description: str = Field(
        description="2-3 sentences explaining the insight in plain language, "
        "referencing the real column names and numbers from the data."
    )
    supporting_stat: str = Field(
        description="The concrete statistic backing this up, e.g. "
        "'corr(age, income) = 0.71' or 'churn rate: 42% (month-to-month) vs "
        "11% (annual)'. Must be something computable from the profile given."
    )
    importance: int = Field(
        ge=1,
        le=5,
        description="How important/actionable this insight is: 1 = minor "
        "curiosity, 5 = critical finding that should drive a decision.",
    )


class InsightReport(BaseModel):
    insights: List[Insight] = Field(
        description="Between 3 and 7 insights about this dataset, ordered by "
        "importance descending. Quality over quantity -- do not pad with "
        "restated summary statistics that aren't actually insightful."
    )