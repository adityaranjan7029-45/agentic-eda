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


class ChartSpec(BaseModel):
    filename: str = Field(
        description="A short filename for this chart, e.g. 'age_vs_income_scatter.png'. "
        "Letters, numbers, underscores, hyphens only -- no slashes or spaces."
    )
    title: str = Field(description="Short, specific chart title, e.g. 'Income rises with age (r=0.71)'.")
    rationale: str = Field(
        description="One sentence: which insight or pattern this chart supports and why "
        "this chart type is the right way to show it."
    )
    code: str = Field(
        description="Self-contained Python code to draw this ONE chart. You may ONLY use "
        "these pre-provided variables -- do not write any import statements: "
        "`df` (the pandas DataFrame), `plt` (matplotlib.pyplot), `sns` (seaborn), "
        "`pd` (pandas), `np` (numpy), and `save_path` (a string -- the file path to save "
        "to). The code MUST end with exactly these three lines, in order: "
        "plt.tight_layout() then plt.savefig(save_path) then plt.close()."
    )


class VisualizationPlan(BaseModel):
    charts: List[ChartSpec] = Field(
        description="2 to 5 charts, each grounded in a real insight or pattern in the data. "
        "Prefer variety (don't make 5 histograms) and prefer charts that make an insight "
        "visually obvious at a glance."
    )


class CriticVerdict(BaseModel):
    approved: bool = Field(
        description="True only if the report is accurate to the source insights, "
        "specific (not generic filler), and free of invented numbers or claims."
    )
    feedback: str = Field(
        description="If approved=False: specific, actionable instructions for what to fix "
        "in the next draft. If approved=True: a one-sentence confirmation of why it's good."
    )