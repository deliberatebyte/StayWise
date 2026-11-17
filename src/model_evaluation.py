import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

REPORT_DIR = RESULTS_DIR / "classification_reports"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
ROC_DIR = RESULTS_DIR / "roc_curves"

for directory in [REPORT_DIR, CONFUSION_DIR, ROC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "xgboost",
]


def load_data():
    X_train = load_npz(DATA_DIR / "X_train.npz")
    X_test = load_npz(DATA_DIR / "X_test.npz")

    y_train = np.load(DATA_DIR / "y_train.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def save_confusion_matrix(name, y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Canceled", "Canceled"],
        yticklabels=["Not Canceled", "Canceled"],
    )

    plt.title(f"{name.replace('_', ' ').title()} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    plt.savefig(CONFUSION_DIR / f"{name}.png", dpi=300)
    plt.close()


def save_roc_curves(roc_data):
    plt.figure(figsize=(9, 7))

    for name, (y_true, y_prob, auc_score) in roc_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)

        plt.plot(
            fpr,
            tpr,
            label=f"{name.replace('_', ' ').title()} (AUC = {auc_score:.4f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(ROC_DIR / "roc_curves.png", dpi=300)
    plt.close()


def evaluate_models():
    print("Loading preprocessed datasets...")

    X_train, X_test, y_train, y_test = load_data()

    print("Training matrix:", X_train.shape)
    print("Testing matrix:", X_test.shape)

    comparison = []
    roc_data = {}

    for name in MODEL_NAMES:
        print(f"\n{'=' * 55}")
        print(f"Evaluating {name.replace('_', ' ').title()}")
        print("=" * 55)

        model_path = MODEL_DIR / f"{name}.joblib"
        model = joblib.load(model_path)

        start_time = time.time()

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_prob = model.predict_proba(X_train)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]

        train_metrics = calculate_metrics(y_train, train_pred, train_prob)
        test_metrics = calculate_metrics(y_test, test_pred, test_prob)

        elapsed = time.time() - start_time

        print("\nTraining Metrics:")
        for metric, value in train_metrics.items():
            print(f"{metric.capitalize():<12}: {value:.4f}")

        print("\nTesting Metrics:")
        for metric, value in test_metrics.items():
            print(f"{metric.capitalize():<12}: {value:.4f}")

        print("\nClassification Report:")
        report = classification_report(
            y_test,
            test_pred,
            labels=[0, 1],
            target_names=["Not Canceled", "Canceled"],
            zero_division=0,
        )

        print(report)

        with open(REPORT_DIR / f"{name}.txt", "w") as file:
            file.write(str(report))

        save_confusion_matrix(name, y_test, test_pred)

        roc_data[name] = (
            y_test,
            test_prob,
            test_metrics["roc_auc"],
        )

        result: dict[str, str | float] = {"model": name}

        for metric in train_metrics:
            result[f"train_{metric}"] = train_metrics[metric]
            result[f"test_{metric}"] = test_metrics[metric]

        result["accuracy_gap"] = train_metrics["accuracy"] - test_metrics["accuracy"]

        result["evaluation_time_seconds"] = elapsed

        comparison.append(result)

        print(f"Evaluation completed in {elapsed:.2f} seconds")

        del model

    save_roc_curves(roc_data)

    comparison_df = pd.DataFrame(comparison)

    comparison_df.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)

    print(comparison_df.round(4).to_string(index=False))

    print("\nBest model based on testing ROC-AUC:")

    best_model = max(
        comparison,
        key=lambda result: float(result["test_roc_auc"]),
    )

    print(
        f"{best_model['model']} " f"(ROC-AUC: {float(best_model['test_roc_auc']):.4f})"
    )

    print("\nAll evaluation results saved successfully!")


if __name__ == "__main__":
    evaluate_models()
