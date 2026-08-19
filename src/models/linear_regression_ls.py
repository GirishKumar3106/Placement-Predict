import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("src/data")

from ingest import load_and_validate_data


def train_linear_regression_ls():

    print("=" * 50)
    print("--- LAB 4: STANDARD LEAST SQUARES ---")

    # Dataset path
    DATA_PATH = os.path.join(
        "src",
        "data",
        "raw_placement_data.csv"
    )

    # Load dataset through ingestion pipeline
    df = load_and_validate_data(DATA_PATH)

    # Use the actual column names from your dataset
    feature_cols = [
        "cgpa",
        "communication_skill_score"
    ]

    target_col = "salary_package_lpa"

    # Remove rows with missing values
    df_clean = df.dropna(
        subset=feature_cols + [target_col]
    ).copy()

    # Input features
    X_raw = df_clean[feature_cols].values

    # Target
    y = df_clean[target_col].values.reshape(-1, 1)

    N = X_raw.shape[0]

    print(f"Loaded {N} data points.")
    print(f"Input features: {feature_cols}")
    print(f"Target: {target_col}")

    # --------------------------------------------------
    # Create Design Matrix
    # --------------------------------------------------
    #
    # X_design = [1, CGPA, Communication Skill Score]
    #
    # The first column of 1s represents the intercept.

    X_design = np.hstack(
        [
            np.ones((N, 1)),
            X_raw
        ]
    )

    print("\nDesign matrix shape:")
    print(X_design.shape)

    # --------------------------------------------------
    # Standard Least Squares / Normal Equation
    # --------------------------------------------------
    #
    # w = (X^T X)^-1 X^T y

    print("\nCalculating optimal weights...")

    XT_X = np.dot(
        X_design.T,
        X_design
    )

    try:
        XT_X_inv = np.linalg.inv(XT_X)

    except np.linalg.LinAlgError:
        print(
            "Matrix is singular. "
            "Using pseudo-inverse instead."
        )

        XT_X_inv = np.linalg.pinv(XT_X)

    XT_y = np.dot(
        X_design.T,
        y
    )

    w_optimal = np.dot(
        XT_X_inv,
        XT_y
    )

    # --------------------------------------------------
    # Display Parameters
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("--- OPTIMAL MODEL PARAMETERS ---")

    print(
        f"Intercept (w0): "
        f"{w_optimal[0, 0]:.4f}"
    )

    print(
        f"Coefficient for cgpa (w1): "
        f"{w_optimal[1, 0]:.4f}"
    )

    print(
        f"Coefficient for communication_skill_score (w2): "
        f"{w_optimal[2, 0]:.4f}"
    )

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    y_pred = np.dot(
        X_design,
        w_optimal
    )

    # Error function

    E_w = 0.5 * np.sum(
        (y_pred - y) ** 2
    )

    print(
        f"\nMinimized Error (E_w): "
        f"{E_w:.4f}"
    )

    # --------------------------------------------------
    # 3D Regression Plane
    # --------------------------------------------------

    print("\nGenerating 3D regression plane...")

    os.makedirs(
        "reports/figures",
        exist_ok=True
    )

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # Actual data points

    ax.scatter(
        X_raw[:, 0],
        X_raw[:, 1],
        y.ravel(),
        alpha=0.4,
        label="Actual Data"
    )

    # Create mesh grid

    x1 = np.linspace(
        X_raw[:, 0].min(),
        X_raw[:, 0].max(),
        30
    )

    x2 = np.linspace(
        X_raw[:, 1].min(),
        X_raw[:, 1].max(),
        30
    )

    x1_mesh, x2_mesh = np.meshgrid(
        x1,
        x2
    )

    # Regression plane

    y_mesh = (
        w_optimal[0, 0]
        + w_optimal[1, 0] * x1_mesh
        + w_optimal[2, 0] * x2_mesh
    )

    ax.plot_surface(
        x1_mesh,
        x2_mesh,
        y_mesh,
        alpha=0.3
    )

    # Labels

    ax.set_xlabel(
        "CGPA"
    )

    ax.set_ylabel(
        "Communication Skill Score"
    )

    ax.set_zlabel(
        "Salary Package LPA"
    )

    ax.set_title(
        "Linear Regression - Standard Least Squares"
    )

    plt.tight_layout()

    output_path = (
        "reports/figures/"
        "linear_regression_3d_plane.png"
    )

    plt.savefig(
        output_path
    )

    plt.close()

    print(
        f"3D regression plane saved to:\n"
        f"{output_path}"
    )

    print("\n" + "=" * 50)
    print("LAB 4 EXECUTION COMPLETE")
    print("=" * 50)


# ------------------------------------------------------
# IMPORTANT: This actually starts the program
# ------------------------------------------------------

if __name__ == "__main__":
    train_linear_regression_ls()