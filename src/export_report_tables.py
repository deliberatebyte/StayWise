import pandas as pd
from pathlib import Path

output_dir = Path("report_tables")
output_dir.mkdir(exist_ok=True)

# Table 3.1: Model Performance Comparison
performance = pd.DataFrame(
    {
        "Evaluation Metric": [
            "Training Accuracy",
            "Testing Accuracy",
            "Training Precision",
            "Testing Precision",
            "Training Recall",
            "Testing Recall",
            "Training F1-score",
            "Testing F1-score",
            "Training ROC-AUC",
            "Testing ROC-AUC",
        ],
        "Logistic Regression": [
            "80.11%",
            "79.73%",
            "79.11%",
            "78.68%",
            "80.11%",
            "79.73%",
            "79.18%",
            "78.77%",
            "85.92%",
            "85.54%",
        ],
        "Random Forest": [
            "99.69%",
            "84.39%",
            "99.69%",
            "83.88%",
            "99.69%",
            "84.39%",
            "99.69%",
            "83.90%",
            "99.99%",
            "90.29%",
        ],
        "XGBoost": [
            "85.25%",
            "84.18%",
            "84.85%",
            "83.74%",
            "85.25%",
            "84.18%",
            "84.92%",
            "83.85%",
            "92.12%",
            "90.88%",
        ],
    }
)

# Table 3.2: Logistic Regression
logistic = pd.DataFrame(
    [
        ["Not Canceled", 0.83, 0.91, 0.87, 12675],
        ["Canceled", 0.67, 0.51, 0.58, 4805],
        ["Accuracy", "", "", 0.80, 17480],
        ["Macro Avg", 0.75, 0.71, 0.72, 17480],
        ["Weighted Avg", 0.79, 0.80, 0.79, 17480],
    ],
    columns=["Class", "Precision", "Recall", "F1-score", "Support"],
)

# Table 3.3: Random Forest
random_forest = pd.DataFrame(
    [
        ["Not Canceled", 0.87, 0.93, 0.90, 12675],
        ["Canceled", 0.76, 0.63, 0.69, 4805],
        ["Accuracy", "", "", 0.84, 17480],
        ["Macro Avg", 0.81, 0.78, 0.79, 17480],
        ["Weighted Avg", 0.84, 0.84, 0.84, 17480],
    ],
    columns=["Class", "Precision", "Recall", "F1-score", "Support"],
)

# Table 3.4: XGBoost
xgboost = pd.DataFrame(
    [
        ["Not Canceled", 0.87, 0.91, 0.89, 12675],
        ["Canceled", 0.74, 0.65, 0.69, 4805],
        ["Accuracy", "", "", 0.84, 17480],
        ["Macro Avg", 0.81, 0.78, 0.79, 17480],
        ["Weighted Avg", 0.84, 0.84, 0.84, 17480],
    ],
    columns=["Class", "Precision", "Recall", "F1-score", "Support"],
)

# Export all tables
tables = {
    "Table_3.1_Model_Comparison": performance,
    "Table_3.2_Logistic_Regression": logistic,
    "Table_3.3_Random_Forest": random_forest,
    "Table_3.4_XGBoost": xgboost,
}

for filename, dataframe in tables.items():
    dataframe.to_csv(output_dir / f"{filename}.csv", index=False)

# Also create one Excel workbook containing all tables
with pd.ExcelWriter(output_dir / "StayWise_Evaluation_Tables.xlsx") as writer:
    for sheet_name, dataframe in tables.items():
        dataframe.to_excel(writer, sheet_name=sheet_name[:31], index=False)

print("All report tables generated successfully!")
print(f"Output directory: {output_dir.resolve()}")
