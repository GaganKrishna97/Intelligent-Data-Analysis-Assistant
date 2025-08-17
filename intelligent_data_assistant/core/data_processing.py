import pandas as pd

def load_dataset(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format")

    # Safely try datetime and numeric conversion
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception:
            pass
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
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

    # Additional insights/examples
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        profile["suggestions"].append(f"Columns with only one unique value (constant): {', '.join(constant_cols)}")
    
    # Default/fallback message if no suggestions found
    if not profile["suggestions"]:
        profile["suggestions"].append(
            "No major issues detected. The data appears clean and ready for analysis."
        )
    return profile
