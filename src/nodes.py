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
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

def type_conversion_node(state: GraphState):
    print("-> Executing typr Conversion...")

    df=state["df"].copy()
    plan=state["plan"]

    bool_mapping={
        'yes': True, 'no': False,
        'true': True, 'false': False,
        't': True, 'f': False,
        '1': True, '0':False
    }

    obj_cols=df.select_dtypes(incluse=['object', 'string']).columns
    for col in obj_cols:
        #Drop NaNs temporarily just to check the unique values
        unique_vals=df[col].dropna().unique()
        # If the column ONLY contains values from our mapping, convert it
        if len(unique_vals) > 0 and all(str(val).lower() in bool_mapping for val in unique_vals):
            df[col] = df[col].astype(str).str.lower().map(bool_mapping)
            
    # 3. Convert Strings to Numerics
    # Refetch object columns since some might have just become booleans
    obj_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in obj_cols:
        try:
            # errors='ignore' safely skips columns that are actual text (like 'Names')
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except Exception:
            pass
            
    # 4. Convert Strings to Datetimes
    # Look for columns that contain date-like keywords
    date_keywords = ['date', 'time', 'year', 'month', 'day', 'timestamp']
    obj_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in obj_cols:
        if any(keyword in col.lower() for keyword in date_keywords):
            try:
                # errors='coerce' turns unparseable garbage dates into NaT (Not a Time)
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass
                
    # 5. Downcast Numerics to Save Memory
    # Converts float64 to float32, and int64 to int32/int8 to make the dataframe lighter
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')
        
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    # 6. Update Graph State
    completed_step = plan.steps.pop(0)
    print(f"   [+] Completed: {completed_step}. Data types optimized.")
    
    return {"df": df, "plan": plan}
        

def imputation_node(state: GraphState):
    print("-> Executing Imputation...")

    #1. Extract State
    df= state["df"].copy()
    plan = state["plan"]

    #Calculate the percentage of missing values for every column.

    missing_percentages=(df.isnull().sum() / len(df)) * 100

    for col in df.columns:
        missing_pct = missing_percentages[col]

        # If the column has ANY missing values, we process it
        if missing_pct > 0:

            #2. Threshold Check
            #If a column is mostly empty, imputing it creates fake data. we drop it.
            if missing_pct >30.0:
                print(f" [!] Dropping column '{col}' (>30% missing: {missing_pct:.1f}%)")
                df.drop(columns=[col],inplace=True)

            #3. Numeric Imputation (Mean/Median)
            elif pd.api.types.is_numeric_dtupe(df[col]):

                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

            #4. Categorical Imputation (Mode)

            else:
                #For strings/objects, We fill with the most frequent value (mode)
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)

    # 5. Update Graph State
    completed_step = plan.steps.pop(0)
    print(f" [+] Completed: {completed_step}. Missing Values Successfully handled.")

    return {"df": df, "plan": plan}

def outlier_handling_node(state: GraphState):
    print("-> Executing Outlier Handling...")
    
    # 1. Extract state (Indented 4 spaces!)
    df = state["df"].copy()
    plan = state["plan"]
    
    # 2. Isolate numerical columns
    # We only handle outliers for continuous numerical data, not categories or dates
    num_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    
    for col in num_cols:
        # 3. Calculate the 1st and 99th percentiles
        # This ignores the absolute craziest 1% of data on both ends
        lower_bound = df[col].quantile(0.01)
        upper_bound = df[col].quantile(0.99)
        
        # 4. Cap (Clip) the values
        # Anything below the lower bound becomes the lower bound.
        # Anything above the upper bound becomes the upper bound.
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    # 5. Update Graph State
    completed_step = plan.steps.pop(0)
    print(f"   [+] Completed: {completed_step}. Extreme values capped at 1st/99th percentiles.")
    
    return {"df": df, "plan": plan}

            
def feature_engineering_node(state: GraphState):
    print("-> Executing Universal Feature Engineering...")
    
    df = state["df"].copy()
    plan = state["plan"]
    
    # ---------------------------------------------------------
    # 1. TEMPORAL & TEXT FEATURES (From previous step)
    # ---------------------------------------------------------
    dt_cols = df.select_dtypes(include=['datetime64', 'datetimetz']).columns
    for col in dt_cols:
        df[f"{col}_month_sin"] = np.sin(2 * np.pi * df[col].dt.month / 12.0)
        df[f"{col}_month_cos"] = np.cos(2 * np.pi * df[col].dt.month / 12.0)
        df[f"{col}_is_weekend"] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
        df.drop(columns=[col], inplace=True)
        
    obj_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in obj_cols:
        if df[col].nunique() > 50:
            df[f"{col}_char_count"] = df[col].astype(str).str.len()
            df[f"{col}_word_count"] = df[col].astype(str).apply(lambda x: len(x.split()))

    # ---------------------------------------------------------
    # 2. AUTOMATED BINNING (Discretization)
    # Turns continuous numbers into categories to capture non-linear patterns.
    # ---------------------------------------------------------
    num_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    
    for col in num_cols:
        # Only bin columns that actually have a wide spread of unique values
        if df[col].nunique() > 20:
            try:
                # Cut into 4 quantiles (e.g., 0-25%, 25-50%, etc.)
                # duplicates='drop' prevents crashes if a lot of data is zeroes
                df[f"{col}_binned"] = pd.qcut(df[col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
            except Exception:
                pass # Fail gracefully if column distribution is too skewed

    # ---------------------------------------------------------
    # 3. CONTROLLED POLYNOMIAL INTERACTIONS
    # Only interact the top highly-variant numeric columns to prevent explosions
    # ---------------------------------------------------------
    # Select up to 10 numerical columns with the highest variance
    high_var_cols = df[num_cols].var().nlargest(10).index.tolist()
    
    # Create simple pair-wise ratios (Col_A / Col_B)
    for i in range(len(high_var_cols)):
        for j in range(i + 1, len(high_var_cols)):
            col_a = high_var_cols[i]
            col_b = high_var_cols[j]
            
            # Create Ratio (Add 1e-5 to prevent Divide by Zero errors)
            ratio_name = f"{col_a}_div_{col_b}"
            df[ratio_name] = df[col_a] / (df[col_b] + 1e-5)

    # ---------------------------------------------------------
    # 4. State Update
    # ---------------------------------------------------------
    completed_step = plan.steps.pop(0)
    print(f"   [+] Completed: {completed_step}. Dataset expanded from {len(state['df'].columns)} to {len(df.columns)} columns.")
    
    return {"df": df, "plan": plan}

def encoding_node(state: GraphState):
    print("-> Executing Categorical Encoding...")
    
    # 1. Extract state
    df = state["df"].copy()
    plan = state["plan"]
    
    # 2. Identify all categorical/text columns
    # We look for objects, strings, and pandas 'category' types
    categorical_cols = df.select_dtypes(include=['object', 'string', 'category']).columns
    
    for col in categorical_cols:
        num_unique = df[col].nunique()
        
        # 3. LOW CARDINALITY: One-Hot Encoding
        # If there are fewer than 15 unique categories (e.g., 'Color': Red, Blue, Green)
        if num_unique < 15:
            # We use drop_first=True to prevent the "Dummy Variable Trap" (Multicollinearity).
            # For example, if a column is Male/Female, we only need an 'is_Male' column.
            # If is_Male is 0, the model mathematically deduces Female.
            df = pd.get_dummies(df, columns=[col], drop_first=True)
            print(f"   [+] One-Hot Encoded '{col}' ({num_unique} unique values)")
            
        # 4. HIGH CARDINALITY: Label / Ordinal Encoding
        # If there are many categories (e.g., 'Zipcode', 'City', or 'Job Title')
        # One-Hot Encoding would explode the dataset dimensions. We convert them to integers.
        else:
            # pd.factorize assigns a unique integer to each unique string
            # e.g., "New York" -> 0, "London" -> 1, "Tokyo" -> 2
            df[col] = pd.factorize(df[col])[0]
            print(f"   [+] Label Encoded '{col}' ({num_unique} unique values)")
            
    # 5. BOOLEAN SAFETY CHECK
    # Machine learning models (like XGBoost or older Scikit-Learn versions) 
    # will crash if they see 'True/False' strings or pure booleans. 
    # We must convert any boolean columns to 1 and 0.
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

    # 6. Update Graph State
    completed_step = plan.steps.pop(0)
    print(f"   [+] Completed: {completed_step}. Dataset is now 100% numerical (Shape: {df.shape}).")
    
    return {"df": df, "plan": plan}


def feature_transformation_node(state: GraphState):
    print("-> Executing Feature Transformation...")
    
    # 1. Extract state
    df = state["df"].copy()
    plan = state["plan"]
    
    # 2. Identify continuous numerical columns
    # We only want to transform true continuous numbers, not 0/1 boolean columns from encoding.
    num_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    cols_to_transform = [col for col in num_cols if df[col].nunique() > 2]
    
    transformed_count = 0
    
    for col in cols_to_transform:
        # 3. Check Skewness
        # A perfectly normal bell curve has a skew of 0. 
        # Anything > 1 or < -1 is considered highly skewed.
        skewness = df[col].skew()
        
        if abs(skewness) > 1.0:
            try:
                # 4. Apply Yeo-Johnson Power Transformation
                # This mathematically squishes the long tail to create a bell curve.
                pt = PowerTransformer(method='yeo-johnson')
                
                # fit_transform expects a 2D array, so we pass df[[col]]
                df[col] = pt.fit_transform(df[[col]])
                transformed_count += 1
            except Exception:
                # If a specific column fails mathematically, safely skip it
                pass

    # 5. Update Graph State
    completed_step = plan.steps.pop(0)
    print(f"   [+] Completed: {completed_step}. Reshaped {transformed_count} highly skewed columns into bell curves.")
    
    return {"df": df, "plan": plan}

def scaling_node(state: GraphState):
    print("-> Executed Scaling...")

    # 1. Extract state
    df=state["df"].copy()
    plan=state["plan"]

    # 2. Isolate continuous numerical columns.
    # We look for all numbers, but we Must filter out boolean/dummy variables.\

    num_cols=df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns

    cols_to_scale = []
    for col in num_cols:
        # A true continuous feature (like Age or Salary) will have many unique values.
        # If a column has 2 unique values (e.g., 0 and 1), it is an encoded category.
        # We leave the 0s and 1s completely alone.

        if df[col].nunique() > 2:
            cols_to_scale.append(col)
    
    # 3. Apply Standard Scaling (Z-score Normalization)

    if len(cols_to_scale) > 0:
        scaler = StandardScaler()

        #fit_transform calculates the mean and variance, then scales the data.
        #It returns a raw numpy matrix, so we carefully overwrite only the specific columns in our dataframe.

        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    # 4. Update Graph State
    completed_step = plan.steps.pop(0)
    print(f"   [+] Completed: {completed_step}. Standardized {len(cols_to_scale)} continuous features.")
    
    return {"df": df, "plan": plan}


def dimensionality_reduction_node(state: GraphState):
    print("-> Executing Dimensionality Reduction...")

    df = state["df"].copy()
    plan = state["plan"]
    target_col = state.get("target_col" , None) #safely get the target if it exists.

    # 1. Isolate the target variable so PCA doesn't destroy it
    target_data = None
    if target_col and target_col in df.columns:
        target_data = df.pop(target_col)
        print(f"   [!] Isolated target column '{target_col}' from compression.")
    ]
    #2. Dynamic threshold 
    # Only compress if we still have too many feature columns.
    if len(df.columns)>15:
        original_cols = len(df.columns)




        
    

