"""
W3D4: SVM, KNN and Random Forest Comparison

This project:
- Loads the Breast Cancer Wisconsin dataset.
- Trains and tunes SVM and KNN.
- Trains Random Forest.
- Tracks experiments using MLflow with SQLite.
- Saves evaluation results and feature importance.
"""

import os

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
OUTPUT_DIR = "outputs"


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a trained model."""

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)

    print("Confusion Matrix:")
    print(cm)

    result = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    return result, cm


def save_confusion_matrix(cm, model_name):
    """Save confusion matrix as an image."""

    plt.figure(figsize=(5, 4))

    plt.imshow(cm, interpolation="nearest")

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    plt.colorbar()
    plt.tight_layout()

    filename = (
        model_name.lower().replace(" ", "_")
        + "_confusion_matrix.png"
    )

    path = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(path)
    plt.close()

    print(f"Saved: {path}")


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------

    data = load_breast_cancer()

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names,
    )

    y = pd.Series(
        data.target,
        name="target",
    )

    print("Dataset loaded successfully.")
    print(f"Features : {X.shape[1]}")
    print(f"Samples  : {X.shape[0]}")

    # ---------------------------------------------------------
    # 2. Train-test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples : {X_test.shape[0]}")

    # ---------------------------------------------------------
    # 3. MLflow setup
    # ---------------------------------------------------------

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment(
        "W3D4_SVM_KNN_Comparison"
    )

    all_results = []

    # ---------------------------------------------------------
    # 4. SVM
    # ---------------------------------------------------------

    svm_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVC()),
        ]
    )

    svm_params = {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", "auto"],
    }

    print("\nStarting SVM hyperparameter tuning...")

    svm_grid = GridSearchCV(
        svm_pipeline,
        svm_params,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    svm_grid.fit(X_train, y_train)

    print("\nBest SVM parameters:")
    print(svm_grid.best_params_)

    print(
        f"Best SVM CV accuracy: "
        f"{svm_grid.best_score_:.4f}"
    )

    with mlflow.start_run(
        run_name="SVM_Best_Model"
    ):

        mlflow.log_param(
            "model",
            "SVM",
        )

        mlflow.log_params(
            svm_grid.best_params_
        )

        mlflow.log_metric(
            "cv_accuracy",
            svm_grid.best_score_,
        )

        svm_result, svm_cm = evaluate_model(
            svm_grid.best_estimator_,
            X_test,
            y_test,
            "SVM",
        )

        mlflow.log_metrics(
            {
                "test_accuracy": svm_result["accuracy"],
                "test_precision": svm_result["precision"],
                "test_recall": svm_result["recall"],
                "test_f1": svm_result["f1_score"],
            }
        )

        save_confusion_matrix(
            svm_cm,
            "SVM",
        )

        mlflow.log_artifact(
            "outputs/svm_confusion_matrix.png"
        )

    all_results.append(svm_result)

    # ---------------------------------------------------------
    # 5. KNN
    # ---------------------------------------------------------

    knn_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier()),
        ]
    )

    knn_params = {
        "model__n_neighbors": [3, 5, 7, 9],
        "model__weights": [
            "uniform",
            "distance",
        ],
        "model__metric": [
            "euclidean",
            "manhattan",
        ],
    }

    print("\nStarting KNN hyperparameter tuning...")

    knn_grid = GridSearchCV(
        knn_pipeline,
        knn_params,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    knn_grid.fit(X_train, y_train)

    print("\nBest KNN parameters:")
    print(knn_grid.best_params_)

    print(
        f"Best KNN CV accuracy: "
        f"{knn_grid.best_score_:.4f}"
    )

    with mlflow.start_run(
        run_name="KNN_Best_Model"
    ):

        mlflow.log_param(
            "model",
            "KNN",
        )

        mlflow.log_params(
            knn_grid.best_params_
        )

        mlflow.log_metric(
            "cv_accuracy",
            knn_grid.best_score_,
        )

        knn_result, knn_cm = evaluate_model(
            knn_grid.best_estimator_,
            X_test,
            y_test,
            "KNN",
        )

        mlflow.log_metrics(
            {
                "test_accuracy": knn_result["accuracy"],
                "test_precision": knn_result["precision"],
                "test_recall": knn_result["recall"],
                "test_f1": knn_result["f1_score"],
            }
        )

        save_confusion_matrix(
            knn_cm,
            "KNN",
        )

        mlflow.log_artifact(
            "outputs/knn_confusion_matrix.png"
        )

    all_results.append(knn_result)

    # ---------------------------------------------------------
    # 6. Random Forest
    # ---------------------------------------------------------

    print("\nTraining Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)

    with mlflow.start_run(
        run_name="Random_Forest"
    ):

        mlflow.log_param(
            "model",
            "Random Forest",
        )

        mlflow.log_param(
            "n_estimators",
            200,
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE,
        )

        rf_result, rf_cm = evaluate_model(
            rf,
            X_test,
            y_test,
            "Random Forest",
        )

        mlflow.log_metrics(
            {
                "test_accuracy": rf_result["accuracy"],
                "test_precision": rf_result["precision"],
                "test_recall": rf_result["recall"],
                "test_f1": rf_result["f1_score"],
            }
        )

        save_confusion_matrix(
            rf_cm,
            "Random_Forest",
        )

        mlflow.log_artifact(
            "outputs/random_forest_confusion_matrix.png"
        )

    all_results.append(rf_result)

    # ---------------------------------------------------------
    # 7. Feature importance
    # ---------------------------------------------------------

    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": rf.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_importance.csv",
        ),
        index=False,
    )

    print("\nTop 10 Random Forest features:")
    print(
        importance.head(10).to_string(
            index=False
        )
    )

    top_features = (
        importance.head(10)
        .sort_values("importance")
    )

    plt.figure(figsize=(8, 6))

    plt.barh(
        top_features["feature"],
        top_features["importance"],
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(
        "Top 10 Random Forest Feature Importances"
    )

    plt.tight_layout()

    feature_plot = os.path.join(
        OUTPUT_DIR,
        "feature_importance.png",
    )

    plt.savefig(feature_plot)
    plt.close()

    print(f"Saved: {feature_plot}")

    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    # ---------------------------------------------------------
    # 8. Model comparison
    # ---------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    results_df = results_df.sort_values(
        "accuracy",
        ascending=False,
    )

    results_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "model_results.csv",
        ),
        index=False,
    )

    print("\nModel Comparison:")
    print(
        results_df.to_string(
            index=False
        )
    )

    print("\nW3D4 completed successfully.")


if __name__ == "__main__":
    main()