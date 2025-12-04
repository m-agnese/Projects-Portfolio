"""
Train ML model to predict missing CCPI values using XGBoost.

This script performs:
1. Loads preprocessed dataset
2. Preprocessing: imputation, scaling, one-hot encoding
3. Feature selection using SHAP
4. Hyperparameter tuning with Optuna
5. Model training and evaluation
6. Prediction of missing CCPI values

Inputs:
- data_processed/CCPI_DATASET.xlsx (preprocessed dataset)

Outputs:
- results/Predicted_CCPI_Values.xlsx
- models/preprocessor.pkl
- models/selected_features.pkl
- models/final_xgb_ccpi_model.pkl
"""

import os
import pandas as pd
import numpy as np
import joblib
import optuna
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Paths
input_file = os.path.join("data_processed", "CCPI_DATASET.xlsx")
output_file = os.path.join("results", "Predicted_CCPI_Values.xlsx")
models_dir = "models"

# Ensure output directories exist
os.makedirs(os.path.dirname(output_file), exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Load dataset
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Preprocessed dataset '{input_file}' not found. Run preprocess.py first.")

data = pd.read_excel(input_file)

target_col = "CCPI"
if target_col not in data.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset.")

key_col = data["KEY"] if "KEY" in data.columns else None

# Split data into missing and complete
data_missing_ccpi = data[data[target_col].isna()].copy()
data_complete = data.dropna(subset=[target_col]).copy()

# Validation subset
val_size = min(10, len(data_complete))
data_validation = data_complete.sample(n=val_size, random_state=42)
data_complete = data_complete.drop(data_validation.index)

# Features
drop_cols = ["CCPI"]
if "KEY" in data.columns:
    drop_cols.append("KEY")

X = data_complete.drop(columns=drop_cols, errors="ignore")
y = data_complete[target_col]

# ===========================
# Preprocessing
# ===========================
numerical_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

numerical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# ===========================
# Feature Selection via SHAP
# ===========================
temp_model = XGBRegressor(n_estimators=500, max_depth=5, random_state=42)
temp_model.fit(X_train, y_train)

explainer = shap.Explainer(temp_model)
shap_values = explainer(X_train)
feature_importance = np.abs(shap_values.values).mean(axis=0)
important_features = np.argsort(feature_importance)[-15:]

X_train = X_train[:, important_features]
X_test = X_test[:, important_features]

# ===========================
# Hyperparameter Optimization
# ===========================
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 2000, 5000, step=500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, step=0.01),
        "max_depth": trial.suggest_int("max_depth", 5, 7),
        "subsample": trial.suggest_float("subsample", 0.7, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.85),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 8),
        "gamma": trial.suggest_float("gamma", 0.1, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 10, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 10, 20),
    }
    model = XGBRegressor(objective="reg:squarederror", random_state=42, **params)
    return cross_val_score(model, X_train, y_train, cv=7, scoring="r2").mean()

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=75)

# ===========================
# Train Final Model
# ===========================
best_params = study.best_params
final_model = XGBRegressor(objective="reg:squarederror", random_state=42, **best_params)
final_model.fit(X_train, y_train)

# ===========================
# Model Evaluation
# ===========================
y_pred = final_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Evaluation:")
print(f"R2 Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

cv_scores = cross_val_score(final_model, X_train, y_train, cv=7, scoring="r2")
print(f"Mean Cross-Validation R2 Score: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

# ===========================
# Predict Missing CCPI Values
# ===========================
keys_missing = data_missing_ccpi["KEY"] if "KEY" in data_missing_ccpi.columns else None
X_missing = preprocessor.transform(data_missing_ccpi.drop(columns=["CCPI", "KEY"], errors="ignore"))
X_missing = X_missing[:, important_features]
data_missing_ccpi[target_col] = final_model.predict(X_missing)

# Restore KEY column if needed
if keys_missing is not None and "KEY" not in data_missing_ccpi.columns:
    data_missing_ccpi.insert(0, "KEY", keys_missing)

# Save predictions and models
data_missing_ccpi.to_excel(output_file, index=False)
joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.pkl"))
joblib.dump(important_features, os.path.join(models_dir, "selected_features.pkl"))
joblib.dump(final_model, os.path.join(models_dir, "final_xgb_ccpi_model.pkl"))

print(f"\nPredicted CCPI values saved to '{output_file}'.")
print("Preprocessor, selected features, and final model saved in 'models/' folder.")
