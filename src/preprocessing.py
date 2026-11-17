from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import save_npz
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("dataset/hotel_bookings.csv")
OUTPUT_DIR = Path("data/processed")
TARGET = "is_canceled"
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

original_rows = len(df)
df = df.drop_duplicates().copy()
duplicates_removed = original_rows - len(df)

df = df.drop(
    columns=[
        "reservation_status",
        "reservation_status_date",
        "booking_changes",
        "assigned_room_type",
        "company",
    ],
    errors="ignore",
)

X = df.drop(columns=[TARGET])
y = df[TARGET]

numeric_features = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

categorical_features = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "agent",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
]

X[categorical_features] = (
    X[categorical_features]
    .astype("string")
    .astype(object)
    .where(X[categorical_features].notna(), np.nan)
)

numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ],
    sparse_threshold=1.0,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE,
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

save_npz(OUTPUT_DIR / "X_train.npz", X_train_processed)
save_npz(OUTPUT_DIR / "X_test.npz", X_test_processed)

np.save(OUTPUT_DIR / "y_train.npy", y_train.to_numpy())
np.save(OUTPUT_DIR / "y_test.npy", y_test.to_numpy())

joblib.dump(preprocessor, OUTPUT_DIR / "preprocessor.joblib")

feature_names = preprocessor.get_feature_names_out().tolist()

with open(OUTPUT_DIR / "feature_names.json", "w") as file:
    json.dump(feature_names, file)

print("Preprocessing completed successfully!")
print("Original records:", original_rows)
print("Duplicates removed:", duplicates_removed)
print("Records after deduplication:", len(df))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Training matrix shape:", X_train_processed.shape)
print("Testing matrix shape:", X_test_processed.shape)
print("\nTraining target distribution:")
print(y_train.value_counts())
print("\nTesting target distribution:")
print(y_test.value_counts())
print("\nProcessed files saved to:", OUTPUT_DIR)
