# 📊 Agentic EDA

> **A multi-agent, autonomous Exploratory Data Analysis pipeline built with LangGraph and open-source LLMs (via Groq), that plans its own preprocessing, analyzes the cleaned data, and writes up its own findings.**

## 🚀 Overview

**Agentic EDA** ingests a raw CSV, autonomously decides what preprocessing it needs, cleans and transforms it with deterministic pandas/scikit-learn code, then reads the result to find and report the patterns actually worth knowing about — ending in a polished Markdown report, with no human choosing the steps in between.

### A note on "agentic"

The LLMs in this pipeline never write or execute arbitrary code. Each agent's job is to **reason and decide** — which preprocessing steps apply, what patterns matter, how to phrase the findings — while the actual data manipulation runs through hand-written, tested pandas/scikit-learn functions. This is a deliberate design choice: it keeps every agent's output constrained to a fixed, safe vocabulary via structured (Pydantic-schema) responses, so there's no risk of an LLM hallucinating broken code or corrupting the dataframe. The tradeoff is that the pipeline can only do what its node functions already know how to do — it can sequence and select, not invent new operations from scratch.

## 🧠 System Architecture (LangGraph)

The pipeline is a single `StateGraph` where a chain of specialized agents and deterministic nodes pass a shared state (`GraphState`: the dataframe, the plan, insights, the final report, etc.) forward:

```
Planner Agent (LLM)
      │
      ▼
Preprocessing nodes (deterministic pandas/sklearn)
  data_cleaning → type_conversion → imputation → outlier_handling →
  feature_engineering → encoding → feature_transformation → scaling →
  dimensionality_reduction → feature_selection
  (only the steps the Planner chose actually run, in the order it chose)
      │
      ▼
Insight Agent (LLM)
      │
      ▼
Synthesis Agent (LLM)
      │
      ▼
   Final Report (Markdown)
```

1. **Planner Agent** (`src/graph.py :: planner_node`) — profiles the raw dataset (dtypes, nulls, sample rows) and returns a `PreprocessingPlan`: an ordered list of steps, drawn only from a fixed vocabulary (`data_cleaning`, `type_conversion`, `imputation`, `outlier_handling`, `feature_engineering`, `encoding`, `feature_transformation`, `scaling`, `dimensionality_reduction`, `feature_selection`), plus its reasoning for choosing them. It only includes steps the data actually justifies — e.g. it skips `imputation` if there are no nulls.

2. **Preprocessing nodes** (`src/nodes.py`) — one deterministic function per step in the vocabulary above. A conditional-edge router (`route_next`) reads `plan.steps[0]` after every node and dispatches to the matching function, popping the step off as it completes, until the plan is empty. None of this is LLM-generated code — it's fixed, reviewed pandas/scikit-learn logic (whitespace/dtype cleanup, median/mode imputation, percentile-based outlier capping, one-hot/label encoding, Yeo-Johnson transforms, PCA, correlation-based feature pruning, etc.).

3. **Insight Agent** (`src/graph.py :: insight_agent_node`) — once preprocessing is done, this reads a statistical summary of the *cleaned* data (describe(), top correlations, top category values) and returns 3–7 ranked `Insight` objects, each with a title, plain-language description, a concrete supporting statistic, and an importance score (1–5). It's explicitly told not to invent numbers that aren't in the summary it was given.

4. **Synthesis Agent** (`src/graph.py :: synthesis_agent_node`) — takes the Insight Agent's findings plus a record of which preprocessing steps ran, and writes them up as one Markdown report (Overview, Data Preparation Summary, Key Findings, Recommended Next Steps) aimed at a non-technical reader. Unlike the other two agents, this one is *not* forced into a rigid schema — the desired output is flowing prose, so it's a plain LLM call rather than `with_structured_output`.

### Roadmap (not yet built)

- **Visualization Agent** — would decide which charts best support the findings and generate matplotlib/seaborn code for them. This is the one place LLM-written code would actually run, so it needs a proper sandboxed executor first — held back until the rest of the pipeline is solid.
- **Critic Agent** — would review the plan or the report before it ships and could send work back for revision, giving a genuine reflection/self-healing loop at the reasoning level rather than by patching stack traces.

## 🛠️ Tech Stack

* **Orchestration:** LangGraph, LangChain (`v0.3`)
* **LLMs:** Open-source models (Llama 3.x, GPT-OSS) served via **Groq's free API tier** — `src/config.py` also supports switching to Hugging Face's inference router or OpenAI via the `LLM_PROVIDER` env var, but Groq is the default.
* **Structured Output:** Pydantic schemas (`PreprocessingPlan`, `Insight`/`InsightReport` in `src/schemas.py`) force every planning/analysis response into validated, parseable data instead of free text.
* **Data Science Core:** Pandas, Scikit-learn (PowerTransformer, StandardScaler, PCA)
* **UI:** Streamlit (`app/main.py`) — currently a standalone shell; not yet wired to the LangGraph pipeline above.

## ⚙️ Setup

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in a free Groq API key from [console.groq.com](https://console.groq.com/keys):
   ```
   LLM_PROVIDER=groq
   GROQ_API_KEY=your_key_here
   ```
3. Run the pipeline standalone on any CSV:
   ```
   python -m src.graph path/to/your.csv
   ```
   This prints the Planner's chosen steps, runs preprocessing, prints the Insight Agent's findings, and writes the final report to `report.md`.