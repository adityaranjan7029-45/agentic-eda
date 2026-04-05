# 📊 Agentic EDA

> **An autonomous, self-healing Exploratory Data Analysis agent built with LangGraph, OpenAI, and Groq to generate, execute, and debug data science workflows.**

## 🚀 Overview
**Agentic EDA** is an autonomous data science workflow engine. Designed to bridge generative AI with traditional data analysis, this agent ingests raw CSV datasets, formulates an analytical plan, writes and securely executes its own Pandas/Scikit-learn code, and leverages a self-healing loop to debug runtime errors. The final output is a fully synthesized Markdown report complete with actionable insights and data visualizations.

## 🧠 System Architecture (LangGraph)
The agent operates on a cyclical, state-driven workflow:

1. **Planner Node (Powered by Groq):** Rapidly analyzes dataset metadata and formulates a step-by-step EDA and cleaning plan.
2. **Coder Node (Powered by OpenAI):** Translates the analytical plan into executable Python code utilizing `pandas`, `matplotlib`, and `scikit-learn`.
3. **Executor Node:** Safely executes the generated code in an isolated local environment, capturing output and figures.
4. **Self-Healing Loop (Conditional Edge):** If the execution fails (e.g., syntax errors, `KeyError`), the exact traceback is appended to the State, and the workflow routes back to the Coder Node to rewrite and fix the code autonomously.
5. **Synthesis Node:** Compiles the insights aand generated visualizations into a final, human-readable report.

## 🛠️ Tech Stack
* **Orchestration:** LangGraph, LangChain (`v0.3`)
* **LLMs:** OpenAI GPT-4o (Complex Coding), Groq Llama-3 (High-speed Routing/Planning)
* **Data Science Core:** Pandas, Scikit-learn, Matplotlib, Seaborn
* **Knowledge Retrieval (RAG):** ChromaDB / FAISS

