import pandas as pd
import numpy as np
import os
import sqlite3
import io
from pydantic import BaseModel, Field
from typing import List
from typing import TypedDict
import re
from langgraph.graph import StateGraph, END
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
        _, ext =os.path.splitext(file_path)
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

@tool
def profile_dataframe()->str:
    """
    Calling this tool to get a statistical summary of the loaded dataset.
    """
    global df
    if df is None:
        return "Error: No dataset loaded. Please call load_dataset() first."
    
    #1. Get df.info()
    buffer=io.StringIO() # here adding a buffer to store the df.info() output.
    df.info(buf=buffer)
    info_str=buffer.getvalue()
    
    #2. get null summary
    null_counts=df.isnull().sum()
    null_summary = null_counts[null_counts>0]
    info_str += f"\n\nNull values:\n{null_summary.to_string() if len(null_summary) >  0 else 'None'}"

    #3. Add a sample of data for the llm to see what 'objects' actually look like
    info_str += "\n\nSample Data (First 5 rows):\n"
    info_str += df.head().to_string()

    return info_str


#Here in this class i am literally just doing prompt engineering. Just giving clear prompts to the LLM to work.
#The LLM will look at the profile_dataframe() output and decide what to do.
class PreprocessingPlan(BaseModel):
    steps:List[str] =Field(
        description="""Sequence of Preprocessing operations.
        Allowed Values and their scopes:
       1. 'data_cleaning':First line of defense. Stripping whitespace, standarizing text casing, removing special characters/symbols from strings,and dropping exact duplicate rows.
       2. 'type_conversion' :Casting datatypes. Includes parsing raw strings into datetimes, converting 'object' flags(Yes/No) to booleans, and downcasting numeric types for memory efficiency.
       3. 'imputation': Handling missing/null values. Includes statistical fills (mean/median/mode), forward/backward fills for time-series, or creating 'missing_indicator' boolean columns.
       4. 'outlier_handling': Identifying and treating extreme values. Includes Winsorization (capping/clipping at percentiles) or dropping impossible values based on domain logic.
       5. 'feature_engineering': Creating new predictive signals. Includes mathematical interaction terms(Col_A/Col_B, Col_A*Col_B, Col_A-Col_B, etc.),extracting cyclical time features (hour, day of week),sine/cosine transforms),text length/word count extraction, and binning/bucketing continuous variables into categories.
       6. 'encoding': converting categorical strings to machine - readable numbers.Includes One-Hot Encoding for nominal data, Ordinal Encoding for ranked data, or Frequency/Target encoding for high-cardinality features.
       7. 'feature_transformation': Handling non-Gaussian, skewed distributions. Includes Applying Logarithmic, Box-Cox, Yeo-Johnson, or Frequency/Target encoding for high-cardinality features.
       8. 'scaling': Equalizing the magnitude of numeric features. Includes Standard Scaling (Z-score), Min-Max Scaling, or Robust Scaling(using IQR to ignore outliers).
       9. 'dimensionality_reduction': Reducing the number of features. Includes PCA (Principal Component Analysis) for linear reduction or t-SNE/UMAP for non-linear reduction.
       10. 'feature_selection': Final cleanup. Dropping zero-variance columns, removing redundant ID columns, or dropping highly correlated features to prevent multicollinearity.
        '"""
    )
    reasoning: str=Field(
        description=""" Detailed Chain-of-Thought reasoning for the preprocessing steps.
        Explain exactly WHY each step was chosen, referencing specific columns and data patterns seen in the profile. 
        Explain the logical ORDER of operations (e.g., 'We must handle outliers before scaling', or 'We must convert types before extracting date features')."""
    )

    #Defining the memory of my langgraph.

    class GraphState(TypedDict):
        df: pd.DataFrame
        plan:PreprocessingPlan
        

    
    def data_cleaning_node(state:GraphState):
        print("-> Executing Data Cleaning...")
        df = state["df"].copy()
        plan = state["plan"]
        
        # 1. Drop completely empty rows and columns
        # If a row or column is 100% missing data, it offers no signal.
        df.dropna(how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        
        # 2. Standardize Column Names
        # Converts "First Name!" to "first_name". Prevents key errors later.
        def clean_col_name(col_name):
            col = str(col_name).lower().strip()
            col = re.sub(r'[^a-z0-9_]+', '_', col) # Replace non-alphanumerics with '_'
            col = re.sub(r'_+', '_', col)          # Remove double underscores
            return col.strip('_')                  # Strip leading/trailing underscores
        
        df.rename(columns=lambda x: clean_col_name(x), inplace=True)
        
        # 3. Drop exact duplicate rows
        # Retaining identical observations biases model training.
        df.drop_duplicates(inplace=True)
        
        # 4. Standardize Object/String Columns & Expose Hidden Nulls
        obj_cols = df.select_dtypes(include=['object', 'string']).columns
        
        # List of common junk values that are actually missing data
        hidden_nulls = ["", " ", "null", "none", "n/a", "na", "?", "-", "undefined", "unknown"]
        
        for col in obj_cols:
            # A. Convert to string to avoid errors with mixed types
            # B. Strip leading/trailing whitespace
            # C. Convert everything to lowercase for categorical consistency
            df[col] = df[col].astype(str).str.strip().str.lower()
            
            # D. Replace those junk values with actual np.nan
            df[col] = df[col].replace(hidden_nulls, np.nan)
            
            # E. Re-replace empty strings that might have been created by stripping
            df[col] = df[col].replace('', np.nan)
            
            # F. (Optional but recommended) If a column was purely numeric but stored 
            # as a string with commas/dollars (e.g., "$1,000"), stripping text allows
            # the subsequent type_conversion_node to easily cast it to float.
            df[col] = df[col].str.replace('$', '', regex=False).str.replace(',', '', regex=False)

        # 5. Reset index after dropping rows
        df.reset_index(drop=True, inplace=True)

        # Update Graph State
        completed_step = plan.steps.pop(0)
        print(f"   [+] Completed: {completed_step}. Dataset shape is now {df.shape}.")
        
        return {"df": df, "plan": plan}
        

            



        
    

