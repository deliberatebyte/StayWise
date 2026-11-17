import time
import joblib
import numpy as np

from pathlib import Path
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    X_train = load_npz(DATA_DIR / "X_train.npz")
    y_train = np.load(DATA_DIR / "y_train.npy")

    return X_train, y_train


def train_models(X_train, y_train):
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        ),
    }

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()

        model.fit(X_train, y_train)

        elapsed = time.time() - start_time

        model_path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)

        print(f"Training completed in {elapsed:.2f} seconds")
        print(f"Model saved: {model_path}")


def main():
    print("Loading preprocessed training data...")

    X_train, y_train = load_data()

    print("Training matrix:", X_train.shape)
    print("Target shape:", y_train.shape)

    train_models(X_train, y_train)

    print("\nAll models trained and saved successfully!")


if __name__ == "__main__":
    main()
