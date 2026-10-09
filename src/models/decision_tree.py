# ============================================================
# LAB 8: DECISION TREE CLASSIFICATION
# Placement Prediction
#
# Experiments:
# 1. Gini Index
# 2. Entropy
# 3. Cost Complexity Pruning
# 4. Effect of Maximum Tree Depth
#
# Optimized for faster execution on large datasets.
# ============================================================

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


# ------------------------------------------------------------
# 1. PATHS AND SETTINGS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "src" / "data" / "raw_placement_data.csv"

OUT = PROJECT_ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Limits complexity to improve execution time.
MAX_TREE_DEPTH = 15
MIN_SAMPLES_LEAF = 10

# Maximum number of alpha values evaluated during pruning.
MAX_ALPHA_CANDIDATES = 15

# Sample size used to calculate candidate pruning values.
PRUNING_SAMPLE_SIZE = 20000


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

print("=" * 65)
print("LAB 8: DECISION TREE CLASSIFICATION")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at:\n{DATA_PATH}\n"
        "Check that raw_placement_data.csv exists in src/data."
    )

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print(f"Dataset shape: {df.shape}")


# ------------------------------------------------------------
# 3. SELECT FEATURES AND TARGET
# ------------------------------------------------------------

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

required_columns = FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing from the dataset: "
        f"{missing_columns}"
    )

# Remove rows that do not have a target label.
df = df.dropna(subset=[TARGET]).copy()

X = df[FEATURES].copy()
y_raw = df[TARGET].astype(str).str.strip().str.lower()


# ------------------------------------------------------------
# 4. PREPROCESS DATA
# ------------------------------------------------------------

# Convert placement labels to numeric values.
# 0 = Not Placed
# 1 = Placed

target_map = {
    "not placed": 0,
    "placed": 1,
    "no": 0,
    "yes": 1,
    "false": 0,
    "true": 1,
    "0": 0,
    "1": 1,
}

y = y_raw.map(target_map)

if y.isna().any():
    unknown_labels = sorted(y_raw[y.isna()].unique().tolist())

    raise ValueError(
        "Unexpected values found in placement_status: "
        f"{unknown_labels}\n"
        "Update target_map to match the labels in your dataset."
    )

y = y.astype(int)

# Convert college tier labels such as "Tier 1" and "Tier 2"
# into numbers such as 1 and 2.
X["college_tier"] = pd.to_numeric(
    X["college_tier"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False),
    errors="coerce",
)

# Convert the remaining numerical features.
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

    # Replace missing values with the column median.
    median_value = X[column].median()

    if pd.isna(median_value):
        median_value = 0

    X[column] = X[column].fillna(median_value)

# Handle missing branch names.
X["branch"] = X["branch"].fillna("Unknown").astype(str)

# Convert branch categories into numeric indicator columns.
X = pd.get_dummies(
    X,
    columns=["branch"],
    prefix=["branch"],
    dtype=int,
)

print(f"Rows available: {len(X):,}")
print(f"Encoded feature count: {X.shape[1]}")
print("\nTarget distribution:")
print(y.value_counts().rename(index={
    0: "Not Placed",
    1: "Placed",
}))


# ------------------------------------------------------------
# 5. SPLIT DATASET
# ------------------------------------------------------------
# Training set: used to train models.
# Validation set: used to choose tree settings.
# Test set: reserved for final evaluation.

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_trainval,
)

print("\nDataset split:")
print(f"Training samples:   {len(X_train):,}")
print(f"Validation samples: {len(X_val):,}")
print(f"Testing samples:    {len(X_test):,}")


# ------------------------------------------------------------
# 6. EXPERIMENT 1: GINI INDEX
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("EXPERIMENT 1: DECISION TREE USING GINI INDEX")
print("=" * 65)

start_time = time.time()

gini_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=MAX_TREE_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE,
)

gini_model.fit(X_train, y_train)

gini_train_accuracy = accuracy_score(
    y_train,
    gini_model.predict(X_train),
)

gini_val_accuracy = accuracy_score(
    y_val,
    gini_model.predict(X_val),
)

gini_time = time.time() - start_time

print(f"Training accuracy:   {gini_train_accuracy:.4f}")
print(f"Validation accuracy: {gini_val_accuracy:.4f}")
print(f"Tree depth:          {gini_model.get_depth()}")
print(f"Tree nodes:          {gini_model.tree_.node_count}")
print(f"Execution time:      {gini_time:.2f} seconds")

# Save a readable visualization of the first few tree levels.
plt.figure(figsize=(16, 9))

plot_tree(
    gini_model,
    feature_names=X.columns.tolist(),
    class_names=["Not Placed", "Placed"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=7,
)

plt.title("Decision Tree Using Gini Index")
plt.tight_layout()

plt.savefig(
    OUT / "decision_tree_gini.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print("Saved: decision_tree_gini.png")


# ------------------------------------------------------------
# 7. EXPERIMENT 2: ENTROPY
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("EXPERIMENT 2: DECISION TREE USING ENTROPY")
print("=" * 65)

start_time = time.time()

entropy_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=MAX_TREE_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE,
)

entropy_model.fit(X_train, y_train)

entropy_train_accuracy = accuracy_score(
    y_train,
    entropy_model.predict(X_train),
)

entropy_val_accuracy = accuracy_score(
    y_val,
    entropy_model.predict(X_val),
)

entropy_time = time.time() - start_time

print(f"Training accuracy:   {entropy_train_accuracy:.4f}")
print(f"Validation accuracy: {entropy_val_accuracy:.4f}")
print(f"Tree depth:          {entropy_model.get_depth()}")
print(f"Tree nodes:          {entropy_model.tree_.node_count}")
print(f"Execution time:      {entropy_time:.2f} seconds")

plt.figure(figsize=(16, 9))

plot_tree(
    entropy_model,
    feature_names=X.columns.tolist(),
    class_names=["Not Placed", "Placed"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=7,
)

plt.title("Decision Tree Using Entropy")
plt.tight_layout()

plt.savefig(
    OUT / "decision_tree_entropy.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print("Saved: decision_tree_entropy.png")


# ------------------------------------------------------------
# 8. EXPERIMENT 3: COST COMPLEXITY PRUNING
# ------------------------------------------------------------
# Pruning removes tree branches that add little predictive value.
#
# Optimization:
# - Calculate candidate alphas using a sample.
# - Evaluate at most 15 candidate alpha values.
# - Limit each tree's maximum depth and minimum leaf size.

print("\n" + "=" * 65)
print("EXPERIMENT 3: COST COMPLEXITY PRUNING")
print("=" * 65)

start_time = time.time()

sample_size = min(
    PRUNING_SAMPLE_SIZE,
    len(X_train),
)

X_prune = X_train.sample(
    n=sample_size,
    random_state=RANDOM_STATE,
)

y_prune = y_train.loc[X_prune.index]

print(f"Samples used to find pruning values: {sample_size:,}")

# Calculate candidate pruning alphas from the smaller sample.
pruning_path_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=MAX_TREE_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE,
)

path = pruning_path_model.cost_complexity_pruning_path(
    X_prune,
    y_prune,
)

# Select evenly spaced candidate values instead of testing
# every alpha from the pruning path.
alphas = np.unique(
    np.quantile(
        path.ccp_alphas,
        np.linspace(
            0,
            1,
            MAX_ALPHA_CANDIDATES,
        ),
    )
)

print(f"Candidate alpha values to evaluate: {len(alphas)}")

pruning_rows = []

for index, alpha in enumerate(alphas, start=1):
    model = DecisionTreeClassifier(
        criterion="gini",
        ccp_alpha=float(alpha),
        max_depth=MAX_TREE_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    train_accuracy = accuracy_score(
        y_train,
        model.predict(X_train),
    )

    val_accuracy = accuracy_score(
        y_val,
        model.predict(X_val),
    )

    pruning_rows.append({
        "alpha": float(alpha),
        "train_accuracy": train_accuracy,
        "validation_accuracy": val_accuracy,
        "nodes": model.tree_.node_count,
        "depth": model.get_depth(),
    })

    print(
        f"[{index:02d}/{len(alphas):02d}] "
        f"alpha={alpha:.6f} | "
        f"validation_accuracy={val_accuracy:.4f} | "
        f"nodes={model.tree_.node_count}"
    )

pruning_results = pd.DataFrame(pruning_rows)

# Select alpha using validation accuracy, not test accuracy.
best_pruning_row = pruning_results.loc[
    pruning_results["validation_accuracy"].idxmax()
]

best_alpha = float(best_pruning_row["alpha"])

print(f"\nBest CCP alpha: {best_alpha:.8f}")
print(
    "Best validation accuracy: "
    f"{best_pruning_row['validation_accuracy']:.4f}"
)

# Fit the selected pruned model.
best_pruned_model = DecisionTreeClassifier(
    criterion="gini",
    ccp_alpha=best_alpha,
    max_depth=MAX_TREE_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE,
)

best_pruned_model.fit(X_train, y_train)

pruned_test_accuracy = accuracy_score(
    y_test,
    best_pruned_model.predict(X_test),
)

print(f"Pruned tree depth: {best_pruned_model.get_depth()}")
print(f"Pruned tree nodes: {best_pruned_model.tree_.node_count}")
print(f"Final test accuracy: {pruned_test_accuracy:.4f}")
print(f"Pruning experiment time: {time.time() - start_time:.2f} seconds")

# Save the pruning comparison plot.
plt.figure(figsize=(10, 6))

plt.plot(
    pruning_results["alpha"],
    pruning_results["train_accuracy"],
    marker="o",
    label="Training Accuracy",
)

plt.plot(
    pruning_results["alpha"],
    pruning_results["validation_accuracy"],
    marker="o",
    label="Validation Accuracy",
)

plt.xlabel("CCP Alpha")
plt.ylabel("Accuracy")
plt.title("Cost Complexity Pruning")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUT / "decision_tree_ccp.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print("Saved: decision_tree_ccp.png")


# ------------------------------------------------------------
# 9. EXPERIMENT 4: EFFECT OF TREE DEPTH
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("EXPERIMENT 4: EFFECT OF MAXIMUM TREE DEPTH")
print("=" * 65)

depth_rows = []

for depth in range(1, MAX_TREE_DEPTH + 1):
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    train_accuracy = accuracy_score(
        y_train,
        model.predict(X_train),
    )

    val_accuracy = accuracy_score(
        y_val,
        model.predict(X_val),
    )

    depth_rows.append({
        "max_depth": depth,
        "train_accuracy": train_accuracy,
        "validation_accuracy": val_accuracy,
    })

    print(
        f"Depth={depth:02d} | "
        f"Train Accuracy={train_accuracy:.4f} | "
        f"Validation Accuracy={val_accuracy:.4f}"
    )

depth_results = pd.DataFrame(depth_rows)

best_depth_row = depth_results.loc[
    depth_results["validation_accuracy"].idxmax()
]

best_depth = int(best_depth_row["max_depth"])

print(f"\nBest maximum depth: {best_depth}")
print(
    "Best validation accuracy: "
    f"{best_depth_row['validation_accuracy']:.4f}"
)

# Save depth comparison plot.
plt.figure(figsize=(10, 6))

plt.plot(
    depth_results["max_depth"],
    depth_results["train_accuracy"],
    marker="o",
    label="Training Accuracy",
)

plt.plot(
    depth_results["max_depth"],
    depth_results["validation_accuracy"],
    marker="o",
    label="Validation Accuracy",
)

plt.xlabel("Maximum Tree Depth")
plt.ylabel("Accuracy")
plt.title("Effect of Maximum Tree Depth")
plt.xticks(range(1, MAX_TREE_DEPTH + 1))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUT / "decision_tree_depth.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print("Saved: decision_tree_depth.png")


# ------------------------------------------------------------
# 10. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("LAB 8 SUMMARY")
print("=" * 65)

comparison = pd.DataFrame({
    "Model": [
        "Decision Tree - Gini",
        "Decision Tree - Entropy",
        "Decision Tree - CCP Pruned",
    ],
    "Training Accuracy": [
        gini_train_accuracy,
        entropy_train_accuracy,
        accuracy_score(
            y_train,
            best_pruned_model.predict(X_train),
        ),
    ],
    "Validation Accuracy": [
        gini_val_accuracy,
        entropy_val_accuracy,
        best_pruning_row["validation_accuracy"],
    ],
    "Tree Depth": [
        gini_model.get_depth(),
        entropy_model.get_depth(),
        best_pruned_model.get_depth(),
    ],
    "Tree Nodes": [
        gini_model.tree_.node_count,
        entropy_model.tree_.node_count,
        best_pruned_model.tree_.node_count,
    ],
})

print(comparison.to_string(index=False))

print("\nFinal pruned model test accuracy:")
print(f"{pruned_test_accuracy:.4f}")

print("\nAll figures saved in:")
print(OUT)

print("\nLab 8 execution completed successfully.")