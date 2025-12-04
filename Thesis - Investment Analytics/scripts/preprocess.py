"""
Preprocess CCPI dataset and perform missing data imputation using K-Nearest Neighbors (KNN).

This script performs:
1. Group-wise KNN imputation within country groups (based on the "KEY" column)
2. Global KNN imputation for remaining missing values

Inputs:
- data/CCPI_DB.xlsx (original dataset, optional)

Outputs:
- data_processed/CCPI_DATASET.xlsx (fully imputed dataset)
"""

import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# Optional: ace_tools is used only to display the dataframe if available
try:
    import ace_tools as tools
    USE_ACE_TOOLS = True
except ImportError:
    USE_ACE_TOOLS = False

# Paths
input_file = os.path.join("data", "CCPI_DB.xlsx")
output_file = os.path.join("data_processed", "CCPI_DATASET.xlsx")

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Load dataset
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file '{input_file}' not found. Provide dataset in 'data/' folder.")

data = pd.read_excel(input_file)

# Basic checks
if data.empty:
    raise ValueError("Dataset is empty.")
if "KEY" not in data.columns:
    raise ValueError("Dataset must contain 'KEY' column.")

# Extract country names from 'KEY'
data["Country"] = data["KEY"].str.extract(r'(^[^_]+)')

# ===========================
# Step 1: Group-wise KNN Imputation
# ===========================
grouped_data = data.groupby("Country")
final_imputed_groups = []

for country, group in grouped_data:
    numeric_cols = group.select_dtypes(include=[np.number]).columns
    numeric_data = group[numeric_cols]
    non_numeric_data = group.drop(columns=numeric_cols)

    # Remove columns where all values are missing
    all_missing_cols = numeric_data.columns[numeric_data.isnull().all()]
    filtered_numeric_data = numeric_data.drop(columns=all_missing_cols, errors='ignore')

    # KNN imputation
    imputer = KNNImputer(n_neighbors=5)
    imputed_array = imputer.fit_transform(filtered_numeric_data)
    imputed_numeric_data = pd.DataFrame(imputed_array, columns=filtered_numeric_data.columns, index=group.index)

    # Reintroduce all-missing columns
    for col in all_missing_cols:
        imputed_numeric_data[col] = numeric_data[col]

    # Align columns
    imputed_numeric_data = imputed_numeric_data[numeric_cols]

    # Merge with non-numeric columns
    complete_group = pd.concat([imputed_numeric_data, non_numeric_data], axis=1)
    final_imputed_groups.append(complete_group)

# Combine all groups
final_imputed_data = pd.concat(final_imputed_groups, ignore_index=False)
final_imputed_data["KEY"] = data["KEY"]
final_imputed_data = final_imputed_data[data.columns]

# ===========================
# Step 2: Global KNN Imputation
# ===========================
final_numeric_data = final_imputed_data.select_dtypes(include=[np.number])
imputer_global = KNNImputer(n_neighbors=5)
final_imputed_array = imputer_global.fit_transform(final_numeric_data)
final_imputed_numeric_data = pd.DataFrame(
    final_imputed_array, columns=final_numeric_data.columns, index=final_numeric_data.index
)

final_fully_imputed_data = final_imputed_data.copy()
final_fully_imputed_data[final_numeric_data.columns] = final_imputed_numeric_data

# Verify no missing values remain
assert final_fully_imputed_data.isnull().sum().sum() == 0, "There are still missing values in the dataset."

# Save processed dataset
final_fully_imputed_data.to_excel(output_file, index=False)
print(f"Fully imputed dataset saved to '{output_file}'.")

# Optional: display
if USE_ACE_TOOLS:
    tools.display_dataframe_to_user(name="Fully Imputed Dataset Without Missing Values",
                                    dataframe=final_fully_imputed_data)
