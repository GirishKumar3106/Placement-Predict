# ============================================================
# LAB 9: RANDOM FOREST CLASSIFICATION
# Project: Placement Prediction
#
# Experiments:
# 1. Decision Tree vs Random Forest
# 2. Out-of-Bag (OOB) Error
# 3. Effect of Number of Trees
# 4. Effect of Feature Subsampling
#
# Dataset: raw_placement_data.csv
# Target: placement_status
# ============================================================

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ------------------------------------------------------------
# 1. PATHS AND SETTINGS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "src" / "data" / "raw_placement_data.csv"

OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Keep individual trees reasonably small to control runtime.
MAX_DEPTH = 18
MIN_SAMPLES_LEAF = 5

FEATURES = [
    "branch",
    "college_tier",
    "cgpa",
    "backlogs",
    "coding_skill_score",
    "communication_skill_score",
    "internships_count",
    "projects_count",
]

TARGET = "placement_status"


# ------------------------------------------------------------
# 2. LOAD AND PREPROCESS DATA
# ------------------------------------------------------------

def load_data():

    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA}"
        )

    df = pd.read_csv(DATA)

    # Remove accidental spaces from column names.
    df.columns = df.columns.str.strip()

    required = FEATURES + [TARGET]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Keep the columns needed for this experiment.
    df = df[required].dropna(
        subset=[TARGET]
    ).copy()

    X = df[FEATURES].copy()

    # Convert labels into a consistent format.
    labels = (
        df[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("_", " ", regex=False)
        .str.replace("-", " ", regex=False)
    )

    # 0 = Not Placed
    # 1 = Placed
    target_map = {
        "not placed": 0,
        "notplaced": 0,
        "unplaced": 0,
        "no": 0,
        "false": 0,
        "0": 0,
        "0.0": 0,
        "placed": 1,
        "yes": 1,
        "true": 1,
        "1": 1,
        "1.0": 1,
    }

    y = labels.map(target_map)

    if y.isna().any():
        unknown = labels[y.isna()].unique().tolist()

        raise ValueError(
            "Unknown placement_status labels: "
            f"{unknown}"
        )

    y = y.astype(int)

    if y.nunique() != 2:
        raise ValueError(
            "placement_status must contain both classes: "
            "Placed and Not Placed."
        )

    # Convert college tiers such as "Tier 1" into 1.
    X["college_tier"] = (
        X["college_tier"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
    )

    # Convert numerical features to numeric values.
    numeric_columns = [
        "college_tier",
        "cgpa",
        "backlogs",
        "coding_skill_score",
        "communication_skill_score",
        "internships_count",
        "projects_count",
    ]

    for column in numeric_columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

        # Fill missing numeric values with the median.
        median = X[column].median()

        if pd.isna(median):
            median = 0

        X[column] = X[column].fillna(median)

    # Handle missing branch values.
    X["branch"] = (
        X["branch"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    # Convert branch categories into numerical columns.
    X = pd.get_dummies(
        X,
        columns=["branch"],
        dtype=int,
    )

    return X, y


# ------------------------------------------------------------
# 3. EVALUATION METRICS
# ------------------------------------------------------------

def evaluate_model(y_true, y_pred):

    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "F1 Score": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }


def print_metrics(name, metrics):

    print(f"\n{name}")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


# ------------------------------------------------------------
# 4. MAIN EXPERIMENT
# ------------------------------------------------------------

def run():

    print("=" * 65)
    print("LAB 9: RANDOM FOREST CLASSIFICATION")
    print("=" * 65)

    start_time = time.time()

    # Load data.
    X, y = load_data()

    print(f"\nDataset records: {len(X):,}")
    print(f"Encoded features: {X.shape[1]}")

    print("\nTarget distribution:")
    print(
        y.value_counts()
        .rename(index={
            0: "Not Placed",
            1: "Placed",
        })
    )

    # --------------------------------------------------------
    # 5. TRAIN-TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining records:", f"{len(X_train):,}")
    print("Testing records:", f"{len(X_test):,}")

    # --------------------------------------------------------
    # 6. DECISION TREE BASELINE
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("EXPERIMENT 1: DECISION TREE")
    print("=" * 65)

    dt = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
    )

    dt.fit(X_train, y_train)

    dt_predictions = dt.predict(X_test)

    dt_metrics = evaluate_model(
        y_test,
        dt_predictions,
    )

    print_metrics(
        "Decision Tree Test Results",
        dt_metrics,
    )

    print("Tree depth:", dt.get_depth())
    print("Tree nodes:", dt.tree_.node_count)

    # --------------------------------------------------------
    # 7. RANDOM FOREST AND NUMBER OF TREES
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("EXPERIMENT 2: RANDOM FOREST AND OOB ERROR")
    print("=" * 65)

    tree_counts = [10, 25, 50, 100, 200]

    tree_results = []

    # Warm start lets us add trees to the same forest.
    # At each step only the additional trees are trained.
    forest = RandomForestClassifier(
        n_estimators=tree_counts[0],
        criterion="gini",
        max_features="sqrt",
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        bootstrap=True,
        oob_score=True,
        warm_start=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    rf_100_metrics = None
    rf_100_oob_score = None

    for count in tree_counts:

        print(f"\nTraining forest with {count} trees...")

        forest.set_params(
            n_estimators=count
        )

        # OOB estimates can be incomplete for small forests.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Some inputs do not have OOB scores",
            )

            forest.fit(
                X_train,
                y_train,
            )

        predictions = forest.predict(X_test)

        test_accuracy = accuracy_score(
            y_test,
            predictions,
        )

        oob_score = float(forest.oob_score_)
        oob_error = 1 - oob_score

        tree_results.append({
            "Number of Trees": count,
            "OOB Score": oob_score,
            "OOB Error": oob_error,
            "Test Accuracy": test_accuracy,
        })

        print(f"OOB Score:     {oob_score:.4f}")
        print(f"OOB Error:     {oob_error:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # Keep the 100-tree metrics for baseline comparison.
        if count == 100:

            rf_100_metrics = evaluate_model(
                y_test,
                predictions,
            )

            rf_100_oob_score = oob_score

    tree_results_df = pd.DataFrame(
        tree_results
    )

    print("\n" + "-" * 65)
    print("NUMBER OF TREES RESULTS")
    print("-" * 65)

    print(
        tree_results_df.to_string(
            index=False,
            formatters={
                "OOB Score": "{:.4f}".format,
                "OOB Error": "{:.4f}".format,
                "Test Accuracy": "{:.4f}".format,
            },
        )
    )

    # --------------------------------------------------------
    # 8. PLOT: EFFECT OF NUMBER OF TREES
    # --------------------------------------------------------

    fig, ax1 = plt.subplots(
        figsize=(9, 6)
    )

    line1 = ax1.plot(
        tree_results_df["Number of Trees"],
        tree_results_df["OOB Error"],
        marker="o",
        label="OOB Error",
    )

    ax1.set_xlabel("Number of Trees")
    ax1.set_ylabel("OOB Error")
    ax1.set_title(
        "Effect of Number of Trees - Placement Prediction"
    )

    ax2 = ax1.twinx()

    line2 = ax2.plot(
        tree_results_df["Number of Trees"],
        tree_results_df["Test Accuracy"],
        marker="s",
        linestyle="--",
        label="Test Accuracy",
    )

    ax2.set_ylabel("Test Accuracy")

    lines = line1 + line2
    labels = [
        line.get_label()
        for line in lines
    ]

    ax1.legend(
        lines,
        labels,
        loc="best",
    )

    fig.tight_layout()

    fig.savefig(
        OUT / "random_forest_number_of_trees.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "\nSaved: random_forest_number_of_trees.png"
    )

    # --------------------------------------------------------
    # 9. RANDOM FOREST TEST METRICS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("EXPERIMENT 3: DECISION TREE VS RANDOM FOREST")
    print("=" * 65)

    print_metrics(
        "Decision Tree",
        dt_metrics,
    )

    print_metrics(
        "Random Forest - 100 Trees, max_features='sqrt'",
        rf_100_metrics,
    )

    print(
        f"OOB Score: {rf_100_oob_score:.4f}"
    )

    comparison = pd.DataFrame([
        {
            "Model": "Decision Tree",
            **dt_metrics,
        },
        {
            "Model": "Random Forest - 100 Trees",
            **rf_100_metrics,
        },
    ])

    print("\nModel Comparison:")
    print(
        comparison.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1 Score": "{:.4f}".format,
            },
        )
    )

    # --------------------------------------------------------
    # 10. EFFECT OF FEATURE SUBSAMPLING
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("EXPERIMENT 4: FEATURE SUBSAMPLING")
    print("=" * 65)

    # Compare:
    # sqrt  -> considers approximately sqrt(number of features)
    # log2  -> considers approximately log2(number of features)
    # None  -> considers all features at each split

    feature_options = [
        ("sqrt", "sqrt"),
        ("log2", "log2"),
        ("all features", None),
    ]

    feature_results = []

    for label, option in feature_options:

        print(
            f"\nTesting max_features={label}..."
        )

        # Reuse the already-trained 100-tree sqrt result.
        if label == "sqrt":

            metrics = rf_100_metrics

        else:

            model = RandomForestClassifier(
                n_estimators=100,
                max_features=option,
                max_depth=MAX_DEPTH,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                bootstrap=True,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

            model.fit(
                X_train,
                y_train,
            )

            predictions = model.predict(
                X_test
            )

            metrics = evaluate_model(
                y_test,
                predictions,
            )

        feature_results.append({
            "Feature Subsampling": label,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1 Score": metrics["F1 Score"],
        })

        print_metrics(
            f"Random Forest - {label}",
            metrics,
        )

    feature_results_df = pd.DataFrame(
        feature_results
    )

    print("\nFeature Subsampling Comparison:")

    print(
        feature_results_df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1 Score": "{:.4f}".format,
            },
        )
    )

    # --------------------------------------------------------
    # 11. PLOT: FEATURE SUBSAMPLING
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    bars = plt.bar(
        feature_results_df["Feature Subsampling"],
        feature_results_df["Accuracy"],
    )

    plt.ylabel("Test Accuracy")
    plt.xlabel("max_features")
    plt.title(
        "Feature Subsampling - Placement Prediction"
    )

    plt.ylim(0, 1)

    for bar, value in zip(
        bars,
        feature_results_df["Accuracy"],
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.015, 0.98),
            f"{value:.4f}",
            ha="center",
            fontweight="bold",
        )

    plt.tight_layout()

    plt.savefig(
        OUT / "random_forest_feature_subsampling.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nSaved: random_forest_feature_subsampling.png"
    )

    # --------------------------------------------------------
    # 12. FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("LAB 9 COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print("\nDecision Tree Accuracy:",
          f"{dt_metrics['Accuracy']:.4f}")

    print("Random Forest Accuracy (100 trees):",
          f"{rf_100_metrics['Accuracy']:.4f}")

    print("Random Forest OOB Error (100 trees):",
          f"{1 - rf_100_oob_score:.4f}")

    print("\nGenerated figures are saved in:")
    print(OUT)

    print(
        f"\nTotal execution time: "
        f"{time.time() - start_time:.2f} seconds"
    )


# ------------------------------------------------------------
# 13. PROGRAM ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":
    run()