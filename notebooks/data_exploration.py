import pandas as pd
import numpy as np

DATA_PATH = "dataset/hotel_bookings.csv"

df = pd.read_csv(DATA_PATH)


def section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


section("1. DATASET OVERVIEW")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Memory Usage:", round(df.memory_usage(deep=True).sum() / 1024**2, 2), "MB")


section("2. SAMPLE RECORDS")

print("\nFirst 5 records:")
print(df.head())

print("\nLast 5 records:")
print(df.tail())


section("3. COLUMN NAMES")

for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")


section("4. DATA TYPES")

print(df.dtypes)


section("5. MISSING VALUES")

missing = pd.DataFrame(
    {
        "Missing Count": df.isnull().sum(),
        "Missing Percentage": (df.isnull().mean() * 100).round(2),
    }
)

print(
    missing[missing["Missing Count"] > 0].sort_values(
        by="Missing Percentage", ascending=False
    )
)


section("6. DUPLICATE RECORDS")

print("Exact duplicate rows:", df.duplicated().sum())


section("7. TARGET DISTRIBUTION")

if "is_canceled" in df.columns:
    print(df["is_canceled"].value_counts(dropna=False))
    print("\nPercentage distribution:")
    print((df["is_canceled"].value_counts(normalize=True, dropna=False) * 100).round(2))
    print("\nTarget missing values:", df["is_canceled"].isnull().sum())
else:
    print("Target column not found!")


section("8. NUMERICAL FEATURE STATISTICS")

print(df.describe().T.to_string())


section("9. CATEGORICAL FEATURE STATISTICS")

categorical_cols = df.select_dtypes(include=["object", "category"]).columns

for col in categorical_cols:
    print(f"\n{col}")
    print("Unique values:", df[col].nunique(dropna=False))
    print("Top 5 values:")
    print(df[col].value_counts(dropna=False).head())


section("10. UNIQUE VALUE COUNTS")

for col in df.columns:
    print(f"{col}: {df[col].nunique(dropna=False)}")


section("11. NUMERICAL CORRELATION WITH TARGET")

if "is_canceled" in df.columns:
    correlations = (
        df.select_dtypes(include=np.number)
        .corr()["is_canceled"]
        .drop("is_canceled")
        .sort_values(key=abs, ascending=False)
    )

    print(correlations)


section("12. POTENTIAL DATA LEAKAGE")

suspected_columns = ["reservation_status", "reservation_status_date"]

for col in suspected_columns:
    if col in df.columns:
        print(f"WARNING: {col} requires special attention.")

print("Original shape:", df.shape)
print("Exact duplicates:", df.duplicated().sum())

unique_df = df.drop_duplicates()

print("After removing exact duplicates:", unique_df.shape)

print("\nTarget distribution after deduplication:")
print(unique_df["is_canceled"].value_counts())

print("\nDuplicate distribution by target:")
print(df.loc[df.duplicated(keep=False), "is_canceled"].value_counts())

print("\nInspection completed successfully!")
