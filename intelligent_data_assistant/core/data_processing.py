import pandas as pd

def load_dataset(file):
    try:
        if file.name.endswith(".csv"):
            try:
                df = pd.read_csv(file)
            except UnicodeDecodeError:
                df = pd.read_csv(file, encoding='ISO-8859-1')
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        raise ValueError(f"Failed to load file: {e}")

    if df is None or df.empty or len(df.columns) == 0:
        raise ValueError("The uploaded file appears empty or lacks valid columns. Please check your data.")

    # Only try datetime conversion on likely date/time columns
    for col in df.columns:
        if ('date' in col.lower() or 'time' in col.lower()) and df[col].dtype == object:
            try:
                df[col] = pd.to_datetime(df[col], errors='ignore')
            except Exception:
                pass

    # Try to convert object columns to numeric (errors ignored)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except Exception:
                pass

    return df

def detect_column_types(df):
    col_types = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "text": []
    }
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_types["numeric"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_types["datetime"].append(col)
        elif df[col].nunique() < 50:
            col_types["categorical"].append(col)
        else:
            col_types["text"].append(col)
    return col_types

def profile_data(df):
    profile = {
        "shape": df.shape,
        "missing_by_col": df.isnull().sum().sort_values(ascending=False).to_dict(),
        "n_unique": {col: df[col].nunique() for col in df.columns},
        "example_rows": df.head(3).to_dict(orient="records"),
        "correlations": df.corr(numeric_only=True).to_dict(),
        "suggestions": []
    }

    high_missing = [col for col, n in profile["missing_by_col"].items() if n > 0]
    if high_missing:
        profile["suggestions"].append(
            f"Consider handling missing values in: {', '.join(high_missing)}"
        )
    if any(val > 200 for val in profile["n_unique"].values()):
        profile["suggestions"].append("Some columns have very high cardinality; consider encoding or grouping.")
    if len(df) < 50:
        profile["suggestions"].append("Dataset is small; be cautious with statistical conclusions.")
    elif len(df) > 10000:
        profile["suggestions"].append("Dataset is large; consider sampling for faster computation.")

    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        profile["suggestions"].append(f"Columns with only one unique value (constant): {', '.join(constant_cols)}")

    if not profile["suggestions"]:
        profile["suggestions"].append("No major issues detected. Dataset appears clean and ready.")

    return profile
