# 📊 Agentic EDA

> **A multi-agent, autonomous Exploratory Data Analysis pipeline built with LangGraph and open-source LLMs (via Groq), that plans its own preprocessing, analyzes the cleaned data, and writes up its own findings.**

Upload a raw CSV → five specialized agents plan the preprocessing, clean the data, find the patterns worth knowing about, chart them, and write a report — with no human choosing the steps in between.

## ⚡ Quick start

```bash
git clone https://github.com/adityaranjan7029-45/agentic-eda.git
cd agentic-eda
pip install -r requirements.txt

cp .env.example .env          # then paste a free key from console.groq.com/keys
streamlit run app/main.py
```

That's it — the app opens with a **Try the sample dataset** button so you can see a full run without finding a CSV first.

Prefer the CLI?

```bash
python -m src.graph data/raw/sample_customer_churn.csv Churn
```

Prints the Planner's chosen steps, runs preprocessing, prints the Insight Agent's findings, writes charts to `./charts/`, prints the Critic's verdict, and saves the report to `report.md`.

## 🚀 Overview

**Agentic EDA** ingests a raw CSV, autonomously decides what preprocessing it needs, cleans and transforms it with deterministic pandas/scikit-learn code, then reads the result to find and report the patterns actually worth knowing about — ending in a polished Markdown report.

### A note on "agentic"

The LLMs in this pipeline never write or execute arbitrary *data* code. Each agent's job is to **reason and decide** — which preprocessing steps apply, what patterns matter, how to phrase the findings — while the actual data manipulation runs through hand-written, tested pandas/scikit-learn functions. This is a deliberate design choice: it keeps every agent's output constrained to a fixed, safe vocabulary via structured (Pydantic-schema) responses, so there's no risk of an LLM hallucinating broken code or corrupting the dataframe. The tradeoff is that the pipeline can only do what its node functions already know how to do — it can sequence and select, not invent new operations from scratch.

The one exception is chart drawing, which genuinely does execute generated code — and is sandboxed accordingly. See [Security](#-security) below.

## 🧠 System architecture (LangGraph)

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

1. **Planner Agent** (`src/graph.py :: planner_node`) — profiles the raw dataset (dtypes, nulls, sample rows) and returns a `PreprocessingPlan`: an ordered list of steps drawn only from a fixed vocabulary, plus its reasoning. It only includes steps the data actually justifies — e.g. it skips `imputation` if there are no nulls. Any step name it invents outside the vocabulary is silently dropped before routing.

2. **Preprocessing nodes** (`src/nodes.py`) — one deterministic function per step in the vocabulary. A conditional-edge router (`route_next`) reads `plan.steps[0]` after every node and dispatches to the matching function, popping the step off as it completes, until the plan is empty. None of this is LLM-generated — it's fixed, reviewed pandas/scikit-learn logic (whitespace/dtype cleanup, median/mode imputation, percentile-based outlier capping, one-hot/label encoding, Yeo-Johnson transforms, PCA, correlation-based feature pruning).

3. **Insight Agent** (`src/graph.py :: insight_agent_node`) — reads a statistical summary of the *cleaned* data (describe(), top correlations, top category values) and returns 3–7 ranked `Insight` objects, each with a title, plain-language description, a concrete supporting statistic, and an importance score (1–5). Explicitly told not to invent numbers.

4. **Visualization Agent** (`src/graph.py :: visualization_agent_node`) — proposes 2–5 charts grounded in the Insight Agent's findings and writes matplotlib/seaborn code for each, executed through a hardened sandbox (see [Security](#-security)). If a chart's code throws, the real traceback goes back to the LLM to fix — up to 2 attempts — before that chart is skipped. Charts are a bonus; a failure never stops the pipeline.

5. **Synthesis Agent** (`src/graph.py :: synthesis_agent_node`) — takes the findings, the generated charts, and the record of which preprocessing steps ran, and writes one Markdown report (Overview, Data Preparation Summary, Key Findings, Recommended Next Steps) aimed at a non-technical reader. Unlike the schema-bound agents this is a plain LLM call — the desired output is flowing prose.

6. **Critic Agent** (`src/graph.py :: critic_agent_node`) — reads the finished report back against the Insight Agent's findings (the only source of truth) and returns a `CriticVerdict`. A rejection routes back to Synthesis for a rewrite, capped at 2 revisions so a report the Critic keeps disliking for marginal reasons still ships. This catches a *reasoning* failure — an invented number, vague filler — rather than a code traceback.

## 🔒 Security

The Visualization Agent is the only place LLM-written code executes, so it gets three independent layers:

| Layer | What it stops |
|---|---|
| **Static AST check** (`_assert_chart_code_is_safe`) | Runs *before* `exec`. Rejects imports, any `__dunder__` attribute or name, `eval`/`exec`/`open`/`getattr`/`globals`, and function/class/lambda definitions. |
| **Write jail** (`_make_write_jail`) | `plt.savefig` is proxied; paths are resolved (so `../..` normalizes) and refused outside the charts directory. |
| **Restricted builtins** + `SIGALRM` timeout | Backstop if anything slips the first layer; kills runaway loops. |

**Why the AST layer exists.** A restricted `__builtins__` dict is *not* a sandbox on its own. Removing `__import__` doesn't stop generated code reaching the real interpreter, because every Python object carries a path back to it through its dunder attributes:

```python
().__class__.__bases__[0].__subclasses__()   # → every loaded class, several exposing real builtins
```

That escape was verified working against an earlier version of this sandbox, reaching `os` with root privileges. Separately, `plt.savefig("/anywhere")` survived every builtins restriction, because matplotlib is an object the sandbox legitimately provides and it already holds filesystem access. Both are covered now, and both are kept as regression tests.

This matters because it's *reachable*, not theoretical: uploaded CSV column names flow into the prompt that generates the code that gets executed, which makes a crafted column name a prompt-injection vector ending in code execution.

```bash
pip install pytest
python -m pytest tests/test_sandbox.py -v    # 30 tests
```

Every attack case in that suite was confirmed working *before* the fix. Note the scope: this is defense against a confused or prompt-injected LLM, not a substitute for OS-level isolation if you ever run genuinely untrusted code — for that you want a container or gVisor.

## 🛠️ Tech stack

* **Orchestration:** LangGraph + `langchain-core` (the `langchain` meta-package is deliberately *not* a dependency — nothing imports it)
* **LLMs:** open-source models (GPT-OSS 20b / 120b) via **Groq's free tier**. `src/config.py` also supports Hugging Face's inference router or OpenAI via the `LLM_PROVIDER` env var.
* **Structured output:** Pydantic schemas in `src/schemas.py` force every planning/analysis/review response into validated data instead of free text.
* **Data science core:** pandas, scikit-learn (PowerTransformer, StandardScaler, PCA)
* **Visualization:** matplotlib, seaborn — only through the sandboxed executor
* **UI:** Streamlit (`app/main.py`), fully wired to the pipeline via `graph.stream()` for real per-node progress

## ⚠️ Rate limits on the free tier

Groq's free tier allows **8,000 tokens/minute**. A wide, one-hot-encoded dataset can reach 100+ columns after feature engineering, and dumping full stats for every column pushed a single request to 13,755 tokens — a hard `413 Request too large`.

Two caps in `src/graph.py` keep prompts bounded by showing only the most relevant columns:

```bash
MAX_DESCRIBE_COLS=25     # columns given full describe() stats
MAX_COLUMNS_LISTED=40    # columns listed to the Visualization Agent
```

These are a **free-tier accommodation, not a quality ceiling**. Correlations are still computed across every column — only the printed tables are capped — so the ranking that drives the insights is unaffected. Raise them (or set them in `.env`) on a paid tier and the agents see proportionally more.

## 🚢 Deploying

### Streamlit Community Cloud (recommended)

1. Push to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pick the repo, set the main file to **`app/main.py`**.
4. **App settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Deploy.

`.streamlit/config.toml` is committed, so the deployed app matches local. Apps sleep after 12h of no traffic and wake on the next visit.

### Alternatives

| | RAM | Sleeps | Setup |
|---|---|---|---|
| **Streamlit Cloud** | ~0.7–2.7 GB | after 12h | point at repo |
| **Hugging Face Spaces** | 16 GB, 2 vCPU | on inactivity | Space + config header |
| **Render / Railway** | ~512 MB free | yes, slow cold starts | Dockerfile / start command |

This pipeline is LLM-bound, not RAM-bound, so Streamlit Cloud's lower memory is not a constraint.

### A note on the shared API key

The app resolves keys in this order: **a key the visitor pastes in the sidebar** (session-only, never stored) → **your `GROQ_API_KEY` secret**. If you deploy *with* your key, every visitor shares your 8,000 TPM budget and two concurrent users will rate-limit each other. Deploying **without** a secret is a perfectly good option — the UI cleanly prompts visitors for their own free key.

## 📁 Project layout

```
agentic-eda/
├── app/main.py                    Streamlit UI
├── src/
│   ├── graph.py                   LangGraph wiring, all 5 agents, chart sandbox
│   ├── nodes.py                   deterministic preprocessing functions
│   ├── schemas.py                 Pydantic output schemas
│   └── config.py                  LLM provider switching
├── tests/test_sandbox.py          sandbox security regression tests
├── data/raw/                      sample dataset (uploads gitignored)
├── .streamlit/
│   ├── config.toml                theme + upload limits (committed)
│   └── secrets.toml.example       template — real secrets.toml is gitignored
├── .env.example
└── requirements.txt               pinned
```

## 🔐 Data privacy

Column names, summary statistics and a few literal sample rows are sent to Groq's API to generate the analysis. Don't upload personal, medical, financial or otherwise confidential data. Uploads are processed in memory and not retained after the session ends.