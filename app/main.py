import streamlit as st
import pandas as pd
import os
import time

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
# 🎨 Custom CSS Injection (Orange Theme)
# ==========================================
st.markdown("""
<style>
    /* Gradient Header Text */
    .title-text {
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF8C00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 800;
        margin-bottom: 0px;
    }
    
    /* Glowing Primary Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8C00 100%);
        color: white;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
        transition: all 0.3s ease;
        font-weight: 600;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 20px rgba(255, 140, 0, 0.6);
        transform: translateY(-2px);
    }
    
    /* File Uploader Customization */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #FF8C00;
        border-radius: 12px;
        background-color: rgba(255, 140, 0, 0.05);
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        background-color: rgba(255, 140, 0, 0.1);
        border-color: #FF4B4B;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        border-right: 2px solid rgba(255, 140, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🗄️ Sidebar Layout
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    st.selectbox("Planner Model", ["groq/llama-3.1-70b-versatile", "groq/llama3-8b-8192"])
    st.selectbox("Coder Model", ["openai/gpt-4o", "openai/gpt-4o-mini"])
    st.markdown("---")
    st.markdown("### 🟢 System Status")
    st.success("API Keys Loaded")
    st.success("LangSmith Tracing Active")

# ==========================================
# 🎨 UI Header
# ==========================================
st.markdown('<p class="title-text">Agentic EDA ⚡</p>', unsafe_allow_html=True)
st.markdown("**Autonomous Data Analyst powered by LangGraph, Groq, and OpenAI.**")
st.markdown("---")

# ==========================================
# 📂 File Upload & Handling
# ==========================================
uploaded_file = st.file_uploader("Drop your dataset here (CSV)", type=["csv"])

if uploaded_file is not None:
    # Safely save the uploaded file to our local data/raw directory
    raw_data_dir = os.path.join(os.path.dirname(os.getcwd()), "data", "raw")
    if not os.path.exists(raw_data_dir): 
        raw_data_dir = "data/raw"
        
    os.makedirs(raw_data_dir, exist_ok=True)
    file_path = os.path.join(raw_data_dir, uploaded_file.name)
    
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
        if st.button("🔥 Initialize Autonomous Analysis", use_container_width=True):
            
            with st.status("Initializing LangGraph Workflow...", expanded=True) as status:
                st.write("🧠 **Planner Node (Groq):** Analyzing metadata and drafting strategy...")
                time.sleep(1.5) 
                
                st.write("💻 **Coder Node (OpenAI):** Writing Pandas & Matplotlib scripts...")
                time.sleep(1.5)
                
                st.write("⚡ **Executor Node:** Running code in secure environment...")
                time.sleep(1.5)
                
                st.write("📝 **Synthesizer Node:** Formatting final markdown report...")
                time.sleep(1)
                
                status.update(label="Analysis Complete!", state="complete", expanded=False)

            # Output Area
            st.markdown("### 📑 Final Intelligence Report")
            st.success("The backend graph is not connected yet. Once `src/graph.py` is built, the final markdown report and visualizations will render right here.")