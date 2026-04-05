import streamlit as st
import pandas as pd
import os
import time

# ==========================================
#  Page Configuration
# ==========================================
st.set_page_config(
    page_title="Agentic EDA", 
    page_icon="📊", 
    layout="wide"
)

# ==========================================
#  UI Header
# ==========================================
st.title("📊 Agentic EDA: Autonomous Data Analyst")
st.markdown("""
Welcome to the multi-agent exploratory data analysis pipeline. 
Upload a CSV dataset below, and the LangGraph engine will autonomously plan, write code, execute, and synthesize a final report.
""")

# ==========================================
#  File Upload & Handling
# ==========================================
uploaded_file = st.file_uploader("Drop your dataset here (CSV)", type=["csv"])

if uploaded_file is not None:
    # Safely save the uploaded file to our local data/raw directory
    raw_data_dir = os.path.join(os.path.dirname(os.getcwd()), "data", "raw")
    # Fallback if running from root directory instead of /app
    if not os.path.exists(raw_data_dir): 
        raw_data_dir = "data/raw"
        
    os.makedirs(raw_data_dir, exist_ok=True)
    file_path = os.path.join(raw_data_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File `{uploaded_file.name}` uploaded securely to backend!")

    # Show a quick preview of the data
    try:
        df = pd.read_csv(file_path)
        with st.expander("👀 Preview Raw Data", expanded=False):
            st.dataframe(df.head())
            st.caption(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        st.error(f"Could not read CSV preview: {e}")

    # ==========================================
#  Execution Trigger
# ==========================================
    st.markdown("---")
    if st.button("🚀 Initialize Auto-EDA Agent", type="primary"):
        
        # ⚠️ PLACEHOLDER: This is where we will call `src.graph.invoke()` later
        
        # Simulating the LangGraph Node execution for the UI
        with st.status("Agent Workflow Initiated...", expanded=True) as status:
            st.write("🧠 Planner Node (Groq): Analyzing metadata and drafting strategy...")
            time.sleep(2) 
            
            st.write("💻 Coder Node (OpenAI): Writing Pandas & Matplotlib scripts...")
            time.sleep(2)
            
            st.write("⚡ Executor Node: Running code in secure environment...")
            time.sleep(2)
            
            st.write("📝 Synthesizer Node: Formatting final markdown report...")
            time.sleep(1)
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        #  PLACEHOLDER: Where the final output will go
        st.markdown("### 📑 Final Intelligence Report")
        st.info("The backend graph is not connected yet. Once `src/graph.py` is built, the final markdown report and visualizations will render right here.")

        # Example of the future connection:
        # from src.graph import eda_graph
        # final_state = eda_graph.invoke({"dataset_path": file_path})
        # st.markdown(final_state["report"])
        # st.image(final_state["figures"])