import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from math import sqrt

def simple_auto_ml(df, target_col):
    # Drop rows with missing target
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # One-hot encode non-numeric columns (features)
    X = pd.get_dummies(X, drop_first=True)

    # If target is object/string/categorical, encode as integers
    if y.dtype == object or y.dtype.name == "category":
        y = pd.factorize(y)[0]

    # Regression if more than 20 target values; else classification
    if y.nunique() <= 20 and not pd.api.types.is_float_dtype(y):
        model = RandomForestClassifier(random_state=42)
        task_type = "classification"
    else:
        y = pd.to_numeric(y, errors="raise")
        model = RandomForestRegressor(random_state=42)
        task_type = "regression"

    # Split and fit
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)

    # Score the model
    if task_type == "classification":
        preds = model.predict(X_test)
        score = accuracy_score(y_test, preds)
    else:
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        score = sqrt(mse)  # RMSE, always works

    return model, X_train, score, task_type

# Example usage (assuming df is a pandas DataFrame and 'Proline' is your target):
# model, X_train, score, task_type = simple_auto_ml(df, 'Proline')
