import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the repo root importable regardless of where `streamlit run` was launched from.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SAMPLE_CSV = REPO_ROOT / "data" / "raw" / "sample_customer_churn.csv"

st.set_page_config(
    page_title="Agentic EDA",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colors derive from `currentColor` (the active Streamlit theme's text color), NOT
# from prefers-color-scheme -- Streamlit's theme and the OS theme are independent,
# and keying off the OS one paints white text on a white background.
st.markdown(
    """
<style>
    :root {
        --accent-from: #FF6B35;
        --accent-to:   #F7931E;
        --accent-soft: rgba(255, 107, 53, 0.10);
        --accent-line: rgba(255, 107, 53, 0.30);
    }

    /* Tighten Streamlit's default top padding -- the stock gap is large
       enough that the hero gets pushed below the fold on a laptop. */
    .block-container { padding-top: 2.5rem; max-width: 1180px; }

    /* Hero. The [data-testid] prefix is needed: Streamlit's own `p` rule
       outranks a bare class and reverts the heading to body size. */
    [data-testid="stMarkdownContainer"] p.hero-title {
        font-size: clamp(2.1rem, 4.6vw, 3rem);
        font-weight: 800;
        letter-spacing: -0.025em;
        line-height: 1.08;
        margin: 0 0 0.5rem 0;
        background: linear-gradient(92deg, var(--accent-from), var(--accent-to));
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    [data-testid="stMarkdownContainer"] p.hero-sub {
        font-size: 1.05rem;
        opacity: 0.75;
        margin: 0 0 0.6rem 0;
        max-width: 62ch;
        line-height: 1.55;
    }

    /* ---------- Step cards (empty state) ---------- */
    .steps { display: flex; gap: 0.9rem; flex-wrap: wrap; margin: 0.6rem 0 0.2rem 0; }
    .step {
        flex: 1 1 220px;
        border: 1px solid color-mix(in srgb, currentColor 14%, transparent);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        background: color-mix(in srgb, currentColor 3%, transparent);
    }
    .step-n {
        display: inline-flex; align-items: center; justify-content: center;
        width: 26px; height: 26px; border-radius: 7px;
        background: var(--accent-soft);
        border: 1px solid var(--accent-line);
        font-size: 0.78rem; font-weight: 700;
        color: var(--accent-from);
        margin-bottom: 0.55rem;
    }
    .step-t { font-weight: 650; font-size: 0.95rem; margin-bottom: 0.25rem; }
    .step-d { font-size: 0.85rem; opacity: 0.72; line-height: 1.5; }

    /* ---------- Agent chips ---------- */
    .chips { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-top: 0.2rem; }
    .chip {
        font-size: 0.76rem;
        padding: 0.28rem 0.62rem;
        border-radius: 999px;
        border: 1px solid color-mix(in srgb, currentColor 16%, transparent);
        background: color-mix(in srgb, currentColor 5%, transparent);
        opacity: 0.8;
        white-space: nowrap;
    }

    /* ---------- Buttons ---------- */
    .stButton > button[kind="primary"] {
        background: linear-gradient(92deg, var(--accent-from), var(--accent-to));
        border: none;
        font-weight: 650;
        letter-spacing: 0.01em;
        padding: 0.6rem 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button[kind="primary"]:hover:not(:disabled) {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(255, 107, 53, 0.32);
    }
    /* Disabled Run button must not keep the full-strength gradient. */
    .stButton > button[kind="primary"]:disabled {
        background: color-mix(in srgb, currentColor 12%, transparent);
        color: color-mix(in srgb, currentColor 45%, transparent);
        box-shadow: none;
        transform: none;
    }

    /* ---------- Upload dropzone ---------- */
    [data-testid="stFileUploaderDropzone"] {
        border: 1.5px dashed var(--accent-line);
        border-radius: 12px;
        background: var(--accent-soft);
        transition: border-color 0.2s ease, background 0.2s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent-from); }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] { gap: 0.35rem; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 0.95rem;
        font-weight: 550;
    }

    /* Hide default Streamlit chrome. The Deploy button needs its own rule. */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stAppDeployButton"] { display: none; }
</style>
""",
    unsafe_allow_html=True,
)


# API key: visitor's own (sidebar) wins, else the deployer's (st.secrets / .env).
def _secret(name: str):
    """st.secrets if present, else env. try/except because st.secrets raises
    when no secrets.toml exists."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


OWNER_KEY = _secret("GROQ_API_KEY")


def resolve_api_key(user_key: str):
    """Scope the key to THIS session only. Never os.environ -- that's process-global
    and would leak one visitor's key into another's run (see src/config.py)."""
    from src.config import set_session_api_key

    key = (user_key or "").strip() or OWNER_KEY
    set_session_api_key(key)
    return key


# cache_resource: compile the StateGraph once per process, not per rerun.
# Import inside the function so a missing dep shows an error, not a blank page.
@st.cache_resource
def get_graph():
    from src.graph import build_graph

    return build_graph()


@st.cache_data
def load_sample():
    return pd.read_csv(SAMPLE_CSV)


NODE_MESSAGES = {
    "planner": "🧠 **Planner Agent** — analyzing metadata and drafting a preprocessing plan",
    "data_cleaning": "🧹 Cleaning whitespace, duplicates, and hidden nulls",
    "type_conversion": "🔡 Converting column data types",
    "imputation": "🩹 Filling in missing values",
    "outlier_handling": "📏 Capping extreme outliers",
    "feature_engineering": "🛠️ Engineering new features",
    "encoding": "🔢 Encoding categorical columns",
    "feature_transformation": "📐 Reshaping skewed distributions",
    "scaling": "⚖️ Scaling numeric features",
    "dimensionality_reduction": "📉 Reducing dimensionality (PCA)",
    "feature_selection": "🧬 Dropping redundant features",
    "insight_agent": "🔍 **Insight Agent** — searching the cleaned data for patterns",
    "visualization_agent": "📊 **Visualization Agent** — generating charts",
    "synthesis_agent": "📝 **Synthesis Agent** — writing the final report",
    "critic_agent": "🕵️ **Critic Agent** — reviewing the report for accuracy",
}

AGENTS = ["Planner", "Insight", "Visualization", "Synthesis", "Critic"]

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 🔑 Groq API key")

    user_key = st.text_input(
        "Your Groq API key",
        type="password",
        placeholder="gsk_...",
        help=(
            "Optional. Paste your own free key from console.groq.com to run on "
            "your own rate limit instead of this app's shared one. It's kept in "
            "this browser session only — never stored or logged."
        ),
        label_visibility="collapsed",
    )

    active_key = resolve_api_key(user_key)

    if user_key.strip():
        st.success("Using your key — your own rate limit applies.")
    elif OWNER_KEY:
        st.info("Using the app's shared key. Add your own above for higher limits.")
    else:
        st.error("No key available. Paste one above to run the pipeline.")

    st.caption("[Get a free key →](https://console.groq.com/keys)")

    st.divider()
    st.markdown("### 🤖 Agents")
    st.caption("Models set in `src/graph.py :: AGENT_MODELS`, all served via Groq.")
    st.code(
        "Planner        gpt-oss-20b\n"
        "Insight        gpt-oss-120b\n"
        "Visualization  gpt-oss-120b\n"
        "Synthesis      gpt-oss-20b\n"
        "Critic         gpt-oss-120b",
        language=None,
    )

    st.divider()
    st.caption(
        "Groq's free tier allows 8,000 tokens/min. Very wide datasets get "
        "summarized to fit — raise `MAX_DESCRIBE_COLS` on a paid tier."
    )

# ---------- Hero ----------
st.markdown('<p class="hero-title">Agentic EDA</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Drop in a raw CSV. Five specialized agents plan the preprocessing, '
    "clean the data, find what actually matters, chart it, and write the report — "
    "with no human choosing the steps in between.</p>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="chips">'
    + "".join(f'<span class="chip">{a} Agent</span>' for a in AGENTS)
    + '<span class="chip">LangGraph</span><span class="chip">Groq</span>'
    + "</div>",
    unsafe_allow_html=True,
)

st.write("")

# ---------- Input ----------
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.source_name = None

up_col, sample_col = st.columns([3, 1.15], gap="medium")

with up_col:
    uploaded_file = st.file_uploader("Upload a CSV", type=["csv"], label_visibility="collapsed")

with sample_col:
    st.write("")
    if st.button("Try the sample dataset", width="stretch"):
        st.session_state.df = load_sample()
        st.session_state.source_name = SAMPLE_CSV.name
        st.session_state.pop("result", None)
    st.caption("Customer churn CSV with deliberate missing values.")

# The pipeline sends column names, stats and sample rows to Groq -- say so up front.
st.caption(
    "🔒 **Your data leaves this app.** Column names, summary statistics and a few "
    "sample rows are sent to Groq's API to generate the analysis. Don't upload "
    "personal, medical, financial or otherwise confidential data. "
    "Uploads are processed in memory and not retained after the session ends."
)

if uploaded_file is not None:
    # Persist so a rerun doesn't lose it.
    raw_dir = REPO_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        new_df = pd.read_csv(file_path)
        if st.session_state.source_name != uploaded_file.name:
            st.session_state.pop("result", None)
        st.session_state.df = new_df
        st.session_state.source_name = uploaded_file.name
    except Exception as e:
        st.error(f"Couldn't read that CSV: {e}")
        st.session_state.df = None

df = st.session_state.df

# ---------- Empty state ----------
if df is None:
    st.markdown(
        """
<div class="steps">
  <div class="step">
    <div class="step-n">1</div>
    <div class="step-t">Upload</div>
    <div class="step-d">Any CSV — messy is fine. Missing values, mixed types and
    junk placeholders are what the cleaning nodes are for.</div>
  </div>
  <div class="step">
    <div class="step-n">2</div>
    <div class="step-t">Pick a target (optional)</div>
    <div class="step-d">Name the column you're trying to explain, like
    <code>Churn</code>, and the analysis focuses on what drives it.</div>
  </div>
  <div class="step">
    <div class="step-n">3</div>
    <div class="step-t">Run</div>
    <div class="step-d">The agents plan, preprocess, analyze, chart and
    fact-check — then hand back a report you can download.</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.expander("How it works under the hood"):
        st.markdown(
            """
The LLMs **never write or execute arbitrary data code**. Each agent reasons and
decides — which preprocessing steps apply, what patterns matter, how to phrase a
finding — while the actual transformation runs through hand-written, tested
pandas/scikit-learn functions constrained to a fixed vocabulary.

```
Planner (LLM)
   └─ picks & orders steps from a fixed list
        ↓
Preprocessing nodes (deterministic pandas/sklearn)
   cleaning → types → imputation → outliers → feature engineering →
   encoding → transformation → scaling → PCA → selection
        ↓
Insight Agent (LLM)  →  Visualization Agent (LLM + sandboxed exec)
        ↓
Synthesis Agent (LLM) ⇄ Critic Agent (LLM, rejects up to 2x)
        ↓
Markdown report + PNG charts
```

The one place generated code *does* run — chart drawing — is sandboxed three
ways: a static AST check that rejects imports and dunder access before
anything executes, a restricted builtins table, and a write jail that confines
`savefig` to the charts directory. See `tests/test_sandbox.py`.
"""
        )
    st.stop()

# ---------- Dataset summary + target ----------
st.success(f"Loaded **{st.session_state.source_name}**")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{df.shape[0]:,}")
m2.metric("Columns", f"{df.shape[1]:,}")
m3.metric("Missing cells", f"{int(df.isnull().sum().sum()):,}")
m4.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

with st.expander("Preview raw data"):
    st.dataframe(df.head(20), width="stretch")

NO_TARGET = "(none — general profiling)"
target_choice = st.selectbox(
    "🎯 Target column (optional)",
    [NO_TARGET] + list(df.columns),
    help=(
        "The column you're trying to predict or explain, e.g. 'Churn' or 'Price'. "
        "The Insight Agent will focus on what drives it. Leave blank for general "
        "findings about the dataset instead."
    ),
)
selected_target = target_choice if target_choice != NO_TARGET else None

run_clicked = st.button(
    "⚡ Run the analysis",
    type="primary",
    width="stretch",
    disabled=not active_key,
)

if not active_key:
    st.warning("Add a Groq API key in the sidebar to run the pipeline.")

# ---------- Run ----------
if run_clicked:
    graph = get_graph()
    initial_state = {"df": df}
    if selected_target:
        initial_state["target_col"] = selected_target

    final_state = dict(initial_state)
    pipeline_error = None

    with st.status("Running the pipeline…", expanded=True) as status:
        try:
            # graph.stream() yields one dict per node as it finishes -- real
            # progress, not a fixed sequence of sleeps.
            for step_output in graph.stream(initial_state):
                for node_name, node_result in step_output.items():
                    final_state.update(node_result)
                    st.write(NODE_MESSAGES.get(node_name, f"➡️ {node_name} finished"))
            status.update(label="Analysis complete", state="complete", expanded=False)
        except Exception as e:
            pipeline_error = e
            status.update(label="Pipeline failed", state="error", expanded=True)
            st.write(f"❌ **{type(e).__name__}:** {e}")

    if pipeline_error is not None:
        st.session_state.pop("result", None)
        # Map the failure modes people actually hit to specific advice.
        msg = str(pipeline_error).lower()
        if "rate_limit" in msg or "request too large" in msg or "413" in msg:
            st.error(
                "**Groq rate limit hit.** The free tier allows 8,000 tokens/minute. "
                "Wait a minute and retry, use a narrower dataset, or paste your own "
                "API key in the sidebar."
            )
        elif "api key" in msg or "authentication" in msg or "401" in msg:
            st.error(
                "**API key rejected.** Check the key in the sidebar — get a fresh free "
                "one at console.groq.com/keys."
            )
        else:
            st.error(
                "The pipeline hit an error and couldn't finish. The failure details are "
                "above. Common causes: a rate-limited Groq key, or a CSV shape the "
                "preprocessing steps didn't expect."
            )
    else:
        st.session_state.result = final_state

# ---------- Results ----------
result = st.session_state.get("result")
if result:
    report_md = result.get("report_markdown", "")
    charts = result.get("charts", [])
    final_df = result.get("df")
    insights = result.get("insights", [])
    critic_approved = result.get("critic_approved")
    critic_feedback = result.get("critic_feedback")
    steps_taken = result.get("steps_taken", [])
    resolved_target = result.get("target_col")

    st.divider()

    head_l, head_r = st.columns([3, 1])
    with head_l:
        st.markdown("## Results")
        if resolved_target:
            st.caption(f"Focused on target column **{resolved_target}**")
        else:
            st.caption("General profiling mode — no target column selected.")
    with head_r:
        if critic_approved is True:
            st.success("✅ Critic approved")
        elif critic_approved is False:
            st.warning("⚠️ Shipped after max revisions")

    if critic_feedback:
        with st.expander("Critic's verdict"):
            st.write(critic_feedback)

    tab_report, tab_charts, tab_data, tab_run = st.tabs(
        ["📝 Report", f"📊 Charts ({len(charts)})", "🔬 Cleaned data", "⚙️ Run details"]
    )

    with tab_report:
        if report_md:
            st.markdown(report_md)
            st.download_button(
                "⬇️ Download report (.md)",
                data=report_md,
                file_name="agentic_eda_report.md",
                mime="text/markdown",
            )
        else:
            st.warning("No report was generated.")

    with tab_charts:
        if charts:
            cols = st.columns(2)
            for i, chart in enumerate(charts):
                with cols[i % 2]:
                    if os.path.exists(chart["path"]):
                        st.image(chart["path"], width="stretch")
                        st.caption(f"**{chart['title']}** — {chart.get('rationale', '')}")
                    else:
                        st.warning(f"Chart file missing: {chart['path']}")
        else:
            st.info("No charts were generated for this run.")

    with tab_data:
        if final_df is not None:
            c1, c2 = st.columns(2)
            c1.metric("Final rows", f"{final_df.shape[0]:,}")
            c2.metric("Final columns", f"{final_df.shape[1]:,}")
            st.dataframe(final_df.head(50), width="stretch")
            st.download_button(
                "⬇️ Download cleaned data (.csv)",
                data=final_df.to_csv(index=False).encode("utf-8"),
                file_name="cleaned_dataset.csv",
                mime="text/csv",
            )
            st.caption(
                "Numeric columns are z-score scaled — that's intentional, it's what makes "
                "the output ML-ready. ID-like columns are deliberately left untouched."
            )
        else:
            st.info("No cleaned dataframe in the final state.")

    with tab_run:
        st.markdown("**Preprocessing steps the Planner chose:**")
        if steps_taken:
            st.markdown("\n".join(f"{i}. `{s}`" for i, s in enumerate(steps_taken, 1)))
        else:
            st.caption("No preprocessing was needed.")

        if insights:
            st.markdown("**Raw insights (before the report was written):**")
            for ins in insights:
                st.markdown(
                    f"- **[{ins['importance']}/5] {ins['title']}** — {ins['description']}  \n"
                    f"  _Evidence: {ins['supporting_stat']}_"
                )

        st.caption(f"Critic revisions used: {result.get('critic_revisions', 0)}")