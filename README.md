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
Visualization Agent (LLM + sandboxed code execution)
      │
      ▼
Synthesis Agent (LLM) ◄─────────────┐
      │                              │ rejected (up to 2x)
      ▼                              │
Critic Agent (LLM) ──────────────────┘
      │ approved
      ▼
   Final Report (Markdown) + Charts (PNG)
```

1. **Planner Agent** (`src/graph.py :: planner_node`) — profiles the raw dataset (dtypes, nulls, sample rows) and returns a `PreprocessingPlan`: an ordered list of steps, drawn only from a fixed vocabulary (`data_cleaning`, `type_conversion`, `imputation`, `outlier_handling`, `feature_engineering`, `encoding`, `feature_transformation`, `scaling`, `dimensionality_reduction`, `feature_selection`), plus its reasoning for choosing them. It only includes steps the data actually justifies — e.g. it skips `imputation` if there are no nulls.

2. **Preprocessing nodes** (`src/nodes.py`) — one deterministic function per step in the vocabulary above. A conditional-edge router (`route_next`) reads `plan.steps[0]` after every node and dispatches to the matching function, popping the step off as it completes, until the plan is empty. None of this is LLM-generated code — it's fixed, reviewed pandas/scikit-learn logic (whitespace/dtype cleanup, median/mode imputation, percentile-based outlier capping, one-hot/label encoding, Yeo-Johnson transforms, PCA, correlation-based feature pruning, etc.).

3. **Insight Agent** (`src/graph.py :: insight_agent_node`) — once preprocessing is done, this reads a statistical summary of the *cleaned* data (describe(), top correlations, top category values) and returns 3–7 ranked `Insight` objects, each with a title, plain-language description, a concrete supporting statistic, and an importance score (1–5). It's explicitly told not to invent numbers that aren't in the summary it was given.

4. **Visualization Agent** (`src/graph.py :: visualization_agent_node`) — the one place in the pipeline where an LLM's own code actually runs. It proposes 2–5 charts grounded in the Insight Agent's findings and writes matplotlib/seaborn code for each, executed through a locked-down `exec()` sandbox (`execute_chart_code`): only `df`/`plt`/`sns`/`pd`/`np`/`save_path` are reachable, `__import__` is stripped from the available builtins so no `import` statement can ever succeed, and a `SIGALRM`-based timeout kills runaway code. If a chart's code throws, the real traceback is sent back to the LLM to fix — up to 2 attempts — before that one chart is skipped. This is the self-healing loop from the original concept, deliberately scoped to just chart code rather than the whole pipeline, so a failure can never corrupt the dataframe.

5. **Synthesis Agent** (`src/graph.py :: synthesis_agent_node`) — takes the Insight Agent's findings, the generated charts, and a record of which preprocessing steps ran, and writes them up as one Markdown report (Overview, Data Preparation Summary, Key Findings, Recommended Next Steps) aimed at a non-technical reader. Unlike the schema-bound agents, this one is a plain LLM call rather than `with_structured_output` — the desired output is flowing prose, and forcing it into fixed fields would just mean reassembling Markdown afterwards. On a revision (see below), it also receives the Critic's exact feedback and is told to address it directly.

6. **Critic Agent** (`src/graph.py :: critic_agent_node`) — reads the finished report back against the Insight Agent's findings (the only source of truth) and returns a `CriticVerdict`: approved or not, with specific feedback either way. A rejection routes back to the Synthesis Agent for a rewrite, capped at 2 revisions (`MAX_CRITIC_REVISIONS`) so a report the Critic keeps disliking for marginal reasons still ships rather than looping forever. This is the pipeline's genuine reflection loop — it catches a *reasoning* failure (an invented number, vague filler) rather than a code traceback.

### Not yet done

- **Streamlit wiring** — `app/main.py`'s "Initialize Autonomous Analysis" button still simulates progress with `time.sleep()` calls rather than calling `build_graph().invoke(...)`. The LangGraph pipeline above is fully built and runnable standalone (`python -m src.graph`), just not yet connected to the UI.

## 🛠️ Tech Stack

* **Orchestration:** LangGraph, LangChain (`v0.3`)
* **LLMs:** Open-source models (Llama 3.x, GPT-OSS) served via **Groq's free API tier** — `src/config.py` also supports switching to Hugging Face's inference router or OpenAI via the `LLM_PROVIDER` env var, but Groq is the default.
* **Structured Output:** Pydantic schemas in `src/schemas.py` (`PreprocessingPlan`, `Insight`/`InsightReport`, `ChartSpec`/`VisualizationPlan`, `CriticVerdict`) force every planning/analysis/review response into validated, parseable data instead of free text.
* **Data Science Core:** Pandas, Scikit-learn (PowerTransformer, StandardScaler, PCA)
* **Visualization:** Matplotlib, Seaborn — run only through the sandboxed executor described above, never directly.
* **UI:** Streamlit (`app/main.py`) — currently a standalone shell; not yet wired to the LangGraph pipeline above.

## ⚙️ Setup

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in a free Groq API key from [console.groq.com](https://console.groq.com/keys):
   ```
   LLM_PROVIDER=groq
   GROQ_API_KEY=your_key_here
   ```
3. Run the full pipeline on any CSV:
   ```
   python -m src.graph path/to/your.csv
   ```
   This prints the Planner's chosen steps, runs preprocessing, prints the Insight Agent's findings, generates charts under `./charts/`, prints the Critic's verdict, and writes the final report to `report.md`.