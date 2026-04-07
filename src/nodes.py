import pandas as pd
import os
import sqlite3
from langchain_core.tools import tool

df=None

@tool
def load_dataset(file_path: str)-> str:
    """
    Calling this tool FIRST to load a dataset into memory.
    It supports .csv, .xlsx, .json, and SQLite (.db) databases.
    """
    global df

    try:
        _, ext =os.path.splittext(file_path)
        ext=ext.lower()

        if ext=='.csv':
            df=pd.read_csv(file_path)
        elif ext=='.xlsx':
            df=pd.read_excel(file_path)
        elif ext=='.json':
            df=pd.read_json(file_path)
        elif ext in ['.db', '.sqlite']:
            # For SQL, we connect and grab the first table available
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            if not tables:
                return "Error: No tables found in the database."
                
            table_name = tables[0][0]
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            conn.close()
        else:
            return f"Error: Unsupported file format '{ext}'. Please use csv, xlsx, json, or db."
            # Return a success string to the LLM so it knows it worked
        return f"Successfully loaded data from {file_path}. The dataset has {df.shape[0]} rows and {df.shape[1]} columns."
    except Exception as e:
        # If the file is missing or corrupted, tell the LLM so it doesn't crash
        return f"Error loading data: {str(e)}"