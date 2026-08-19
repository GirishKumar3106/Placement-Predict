import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Make src/data available for import
import sys
sys.path.append("src/data")

from ingest import load_and_validate_data


def compute_cost(X, y, w):
    """
    Compute the cost function:

    J(w) = (1 / 2m) * sum((Xw - y)^2)
    """

    m = len(y)

    predictions = np.dot(X, w)

    errors = predictions - y

    cost = (1 / (2 * m)) * np.sum(errors ** 2)

    return cost


def gradient_descent(X, y, w, alpha, num_iters):
    """
    Gradient Descent implementation using NumPy.
    """

    m = len(y)

    cost_history = []

    for i in range(num_iters):

        # Calculate predictions
        predictions = np.dot(X, w)

        # Calculate errors
        errors = predictions - y

        # Calculate gradient
        gradient = (
            1 / m
        ) * np.dot(
            X.T,
            errors
        )

        # Update weights
        w = w - alpha * gradient

        # Calculate and store cost
        cost = compute_cost(
            X,
            y,
            w
        )

        cost_history.append(cost)

    return w, cost_history


def run_gradient_descent_experiment():

    print("=" * 50)
    print("--- LAB 5: GRADIENT DESCENT LINEAR REGRESSION ---")
    print("=" * 50)

    # --------------------------------------------------
    # 1. Load Dataset
    # --------------------------------------------------

    DATA_PATH = os.path.join(
        "src",
        "data",
        "raw_placement_data.csv"
    )

    df = load_and_validate_data(
        DATA_PATH
    )

    # Lab 5:
    # CGPA -> Salary Package

    feature_col = "cgpa"
    target_col = "salary_package_lpa"

    # Remove missing values
    df_clean = df.dropna(
        subset=[
            feature_col,
            target_col
        ]
    ).copy()

    X = df_clean[
        [feature_col]
    ].values

    y = df_clean[
        [target_col]
    ].values

    print(
        f"Loaded {len(df_clean)} valid samples."
    )

    print(
        f"Feature: {feature_col}"
    )

    print(
        f"Target: {target_col}"
    )

    # --------------------------------------------------
    # 2. Train-Test Split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("\n" + "=" * 50)
    print("--- TRAIN / TEST SPLIT ---")

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # --------------------------------------------------
    # 3. Standardize X and y
    # --------------------------------------------------

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(
        X_train
    )

    X_test_scaled = scaler_X.transform(
        X_test
    )

    y_train_scaled = scaler_y.fit_transform(
        y_train
    )

    y_test_scaled = scaler_y.transform(
        y_test
    )

    # --------------------------------------------------
    # 4. Add Intercept
    # --------------------------------------------------

    X_train_design = np.hstack(
        [
            np.ones(
                (X_train_scaled.shape[0], 1)
            ),
            X_train_scaled
        ]
    )

    X_test_design = np.hstack(
        [
            np.ones(
                (X_test_scaled.shape[0], 1)
            ),
            X_test_scaled
        ]
    )

    print("\nDesign matrix shape:")
    print(
        X_train_design.shape
    )

    # --------------------------------------------------
    # 5. Learning Rate Experiments
    # --------------------------------------------------

    learning_rates = [
        0.001,
        0.01,
        0.1,
        0.5
    ]

    num_iterations = 1000

    results = {}

    os.makedirs(
        "reports/figures",
        exist_ok=True
    )

    plt.figure(
        figsize=(10, 6)
    )

    print("\n" + "=" * 50)
    print("--- LEARNING RATE EXPERIMENTS ---")

    for alpha in learning_rates:

        # Initialize weights to zero

        initial_w = np.zeros(
            (
                X_train_design.shape[1],
                1
            )
        )

        # Run Gradient Descent

        w_optimal, cost_history = gradient_descent(
            X_train_design,
            y_train_scaled,
            initial_w,
            alpha,
            num_iterations
        )

        # Store results

        results[alpha] = {
            "weights": w_optimal,
            "cost_history": cost_history
        }

        final_cost = cost_history[-1]

        print(
            f"Learning rate α = {alpha}"
        )

        print(
            f"Final cost = {final_cost:.6f}"
        )

        # Plot cost history

        plt.plot(
            cost_history,
            label=f"α = {alpha}"
        )

    plt.xlabel(
        "Iterations"
    )

    plt.ylabel(
        "Cost"
    )

    plt.title(
        "Gradient Descent Convergence "
        "for Different Learning Rates"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    output_path = (
        "reports/figures/"
        "gd_learning_rates_comparison.png"
    )

    plt.savefig(
        output_path
    )

    plt.close()

    print(
        f"\nLearning-rate comparison graph saved to:"
    )

    print(
        output_path
    )

    # --------------------------------------------------
    # 6. Select Best Learning Rate
    # --------------------------------------------------

    best_alpha = min(
        results,
        key=lambda alpha:
        results[alpha]["cost_history"][-1]
    )

    best_w = results[
        best_alpha
    ]["weights"]

    print("\n" + "=" * 50)
    print("--- BEST LEARNING RATE ---")

    print(
        f"Best α = {best_alpha}"
    )

    print(
        f"Final training cost = "
        f"{results[best_alpha]['cost_history'][-1]:.6f}"
    )

    # --------------------------------------------------
    # 7. Custom Gradient Descent Parameters
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print(
        "--- CUSTOM GRADIENT DESCENT PARAMETERS ---"
    )

    print(
        f"Intercept (w0): "
        f"{best_w[0, 0]:.6f}"
    )

    print(
        f"Coefficient (w1): "
        f"{best_w[1, 0]:.6f}"
    )

    # --------------------------------------------------
    # 8. Scikit-learn Comparison
    # --------------------------------------------------

    sklearn_model = LinearRegression()

    sklearn_model.fit(
        X_train_scaled,
        y_train_scaled
    )

    print("\n" + "=" * 50)
    print("--- SCIKIT-LEARN COMPARISON ---")

    print(
        f"Scikit-learn Intercept: "
        f"{sklearn_model.intercept_[0]:.6f}"
    )

    print(
        f"Scikit-learn Coefficient: "
        f"{sklearn_model.coef_[0, 0]:.6f}"
    )

    # --------------------------------------------------
    # 9. Parameter Difference
    # --------------------------------------------------

    intercept_difference = abs(
        best_w[0, 0]
        - sklearn_model.intercept_[0]
    )

    coefficient_difference = abs(
        best_w[1, 0]
        - sklearn_model.coef_[0, 0]
    )

    print("\n" + "=" * 50)
    print("--- PARAMETER DIFFERENCE ---")

    print(
        f"Intercept difference: "
        f"{intercept_difference:.6f}"
    )

    print(
        f"Coefficient difference: "
        f"{coefficient_difference:.6f}"
    )

    print("\n" + "=" * 50)
    print("LAB 5 EXECUTION COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    run_gradient_descent_experiment()