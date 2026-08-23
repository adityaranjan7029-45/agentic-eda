import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ==========================================
# 🔧 Make the repo root importable
# ==========================================
# app/main.py lives one level below the repo root. Depending on *how*
# `streamlit run` is launched, the repo root isn't guaranteed to already be
# on sys.path, and `from src.graph import build_graph` below would fail with
# ModuleNotFoundError. This makes it work regardless of the working directory
# the command was run from.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.graph import build_graph  # noqa: E402  (must come after the sys.path fix above)

# ==========================================
# ⚙️ Page Configuration
# ==========================================
st.set_page_config(
    page_title="Agentic EDA",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 🎨 Custom CSS Injection (Animated Orange Theme)
# ==========================================
st.markdown("""
<style>
    /* 1. Page Fade-In Animation */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    .block-container {
        animation: fadeIn 1s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    /* 2. Gradient Header Text with subtle shift */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .title-text {
        background: linear-gradient(-45deg, #FF4B4B, #FF8C00, #FF3366, #FF8C00);
        background-size: 300% 300%;
        animation: gradientShift 6s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5em;
        font-weight: 800;
        margin-bottom: 0px;
    }

    /* 3. Glowing, Breathing Primary Button */
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 10px rgba(255, 140, 0, 0.4); }
        50% { box-shadow: 0 0 25px rgba(255, 75, 75, 0.8); }
        100% { box-shadow: 0 0 10px rgba(255, 140, 0, 0.4); }
    }

    .stButton>button {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8C00 100%);
        color: white;
        border-radius: 8px;
        border: none;
        animation: pulseGlow 2.5s infinite;
        transition: all 0.3s ease;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        animation: none; /* Stops the breathing while hovering */
        box-shadow: 0 8px 25px rgba(255, 140, 0, 0.8);
    }

    /* File Uploader Hover Animation */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #FF8C00;
        border-radius: 12px;
        background-color: rgba(255, 140, 0, 0.05);
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }

    [data-testid="stFileUploadDropzone"]:hover {
        background-color: rgba(255, 140, 0, 0.15);
        border-color: #FF4B4B;
        transform: scale(1.01);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 Cached graph + friendly per-node status text
# ==========================================
# st.cache_resource means build_graph() (which compiles the whole StateGraph)
# only runs once per server process, not on every single Streamlit rerun
# (Streamlit reruns this entire script top-to-bottom on every interaction).
@st.cache_resource
def get_graph():
    return build_graph()


NODE_MESSAGES = {
    "planner": "🧠 **Planner Agent:** Analyzing metadata and drafting a preprocessing plan...",
    "data_cleaning": "🧹 **Preprocessing:** Cleaning whitespace, duplicates, and hidden nulls...",
    "type_conversion": "🔡 **Preprocessing:** Converting column data types...",
    "imputation": "🩹 **Preprocessing:** Filling in missing values...",
    "outlier_handling": "📏 **Preprocessing:** Capping extreme outliers...",
    "feature_engineering": "🛠️ **Preprocessing:** Engineering new features...",
    "encoding": "🔢 **Preprocessing:** Encoding categorical columns...",
    "feature_transformation": "📐 **Preprocessing:** Reshaping skewed distributions...",
    "scaling": "⚖️ **Preprocessing:** Scaling numeric features...",
    "dimensionality_reduction": "📉 **Preprocessing:** Reducing dimensionality (PCA)...",
    "feature_selection": "🧬 **Preprocessing:** Dropping redundant features...",
    "insight_agent": "🔍 **Insight Agent:** Searching the cleaned data for patterns...",
    "visualization_agent": "📊 **Visualization Agent:** Generating charts...",
    "synthesis_agent": "📝 **Synthesis Agent:** Writing the final report...",
    "critic_agent": "🕵️ **Critic Agent:** Reviewing the report for accuracy...",
}

# ==========================================
# 🗄️ Sidebar Layout
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    st.caption("Model per agent (set in src/graph.py :: AGENT_MODELS, all served via Groq)")
    st.text(f"Planner:        {os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')}")
    st.text("Insight:        openai/gpt-oss-120b")
    st.text("Visualization:  openai/gpt-oss-120b")
    st.text("Synthesis:      llama-3.1-8b-instant")
    st.text("Critic:         openai/gpt-oss-120b")
    st.markdown("---")
    st.markdown("### 🟢 System Status")
    if os.getenv("GROQ_API_KEY"):
        st.success("GROQ_API_KEY loaded")
    else:
        st.error("GROQ_API_KEY missing -- add it to your .env")

# ==========================================
# 🎨 UI Header
# ==========================================
st.markdown('<p class="title-text">Agentic EDA ⚡</p>', unsafe_allow_html=True)
st.markdown("**Autonomous, multi-agent Data Analyst powered by LangGraph and Groq.**")
st.markdown("---")

# ==========================================
# 📂 File Upload & Handling
# ==========================================
uploaded_file = st.file_uploader("Drop your dataset here (CSV)", type=["csv"])

if uploaded_file is not None:
    # Save the uploaded file to <repo_root>/data/raw, regardless of cwd.
    raw_data_dir = REPO_ROOT / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_data_dir / uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Create two columns for a cleaner layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.success(f"File `{uploaded_file.name}` uploaded securely to backend!")
        try:
            df = pd.read_csv(file_path)
            with st.expander("👀 Preview Raw Data", expanded=False):
                st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read CSV preview: {e}")
            df = None

    if df is not None:
        with col2:
            st.info(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
            st.info(f"**Memory:** {(df.memory_usage(deep=True).sum() / 1024**2):.2f} MB")

        # ==========================================
        # 🚀 Execution Trigger
        # ==========================================
        st.markdown("---")

        # Using a container to center the button visually
        _, center_col, _ = st.columns([1, 2, 1])
        with center_col:
            run_clicked = st.button("🔥 Initialize Autonomous Analysis", use_container_width=True)

        if run_clicked:
            if not os.getenv("GROQ_API_KEY"):
                st.error("GROQ_API_KEY is not set. Add it to your .env file before running the pipeline.")
            else:
                graph = get_graph()
                initial_state = {"df": df}
                final_state = dict(initial_state)
                pipeline_error = None

                with st.status("Running Agentic EDA pipeline...", expanded=True) as status:
                    try:
                        # graph.stream() yields one dict per node as it finishes,
                        # e.g. {"planner": {"plan": ..., "steps_taken": [...]}} --
                        # this is what lets the UI show REAL progress instead of
                        # a fixed sequence of sleeps that always looks the same
                        # regardless of what the pipeline actually did.
                        for step_output in graph.stream(initial_state):
                            for node_name, node_result in step_output.items():
                                final_state.update(node_result)
                                st.write(NODE_MESSAGES.get(node_name, f"➡️ **{node_name}** finished."))

                        status.update(label="Analysis Complete!", state="complete", expanded=False)
                    except Exception as e:
                        pipeline_error = e
                        status.update(label="Pipeline failed", state="error", expanded=True)
                        st.write(f"❌ {type(e).__name__}: {e}")

                if pipeline_error is not None:
                    st.error(
                        "The pipeline hit an error and couldn't finish. "
                        "See the failure details above. Common causes: an invalid/rate-limited "
                        "Groq API key, or a CSV shape the preprocessing steps didn't expect."
                    )
                else:
                    # ==========================================
                    # 📑 Results
                    # ==========================================
                    report_md = final_state.get("report_markdown", "")
                    charts = final_state.get("charts", [])
                    critic_approved = final_state.get("critic_approved")
                    critic_feedback = final_state.get("critic_feedback")

                    st.markdown("### 📑 Final Intelligence Report")

                    if critic_approved is True:
                        st.success(f"✅ Critic approved this report. {critic_feedback or ''}")
                    elif critic_approved is False:
                        st.warning(
                            f"⚠️ Shipped after the maximum revision attempts -- Critic's last "
                            f"feedback: {critic_feedback or '(none)'}"
                        )

                    if report_md:
                        st.markdown(report_md)
                        st.download_button(
                            "⬇️ Download Report (.md)",
                            data=report_md,
                            file_name="report.md",
                            mime="text/markdown",
                        )
                    else:
                        st.warning("No report was generated.")

                    if charts:
                        st.markdown("### 📊 Charts")
                        chart_cols = st.columns(2)
                        for i, chart in enumerate(charts):
                            with chart_cols[i % 2]:
                                st.image(chart["path"], caption=chart["title"], use_container_width=True)

                    with st.expander("🔬 Final cleaned dataset (preview)", expanded=False):
                        final_df = final_state.get("df")
                        if final_df is not None:
                            st.dataframe(final_df.head(20), use_container_width=True)
                            st.caption(f"Final shape: {final_df.shape[0]} rows, {final_df.shape[1]} columns")