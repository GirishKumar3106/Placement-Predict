import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# ==========================================================
# EXPERIMENT 6: RIDGE / L2 REGULARIZATION
# ==========================================================

# 1. Simulate Data
# CGPA-like input ranging roughly from 4 to 10

np.random.seed(42)

X = np.sort(
    6 * np.random.rand(100, 1) + 4
)

# Non-linear relationship with noise

y = (
    np.sin(X).ravel()
    + np.random.normal(0, 0.2, X.shape[0])
)


# 2. Split dataset into 80% Training and 20% Testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("=" * 60)
print("EXPERIMENT 6: RIDGE / L2 REGULARIZATION")
print("=" * 60)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# 3. Create higher-order polynomial features

poly_degree = 15

poly = PolynomialFeatures(
    degree=poly_degree
)

X_train_poly = poly.fit_transform(
    X_train
)

X_test_poly = poly.transform(
    X_test
)


print(f"Polynomial degree: {poly_degree}")
print(
    f"Polynomial feature count: "
    f"{X_train_poly.shape[1]}"
)


# 4. Scale polynomial features

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_poly
)

X_test_scaled = scaler.transform(
    X_test_poly
)


# 5. Define regularization parameters

lambdas = np.logspace(
    -4,
    4,
    200
)

train_errors = []
test_errors = []


# 6. Train Ridge models for each lambda

for lam in lambdas:

    ridge = Ridge(
        alpha=lam
    )

    ridge.fit(
        X_train_scaled,
        y_train
    )

    # Predictions

    y_train_pred = ridge.predict(
        X_train_scaled
    )

    y_test_pred = ridge.predict(
        X_test_scaled
    )

    # Mean Squared Error

    train_error = mean_squared_error(
        y_train,
        y_train_pred
    )

    test_error = mean_squared_error(
        y_test,
        y_test_pred
    )

    train_errors.append(
        train_error
    )

    test_errors.append(
        test_error
    )


# 7. Find the lambda with minimum test error

best_index = np.argmin(
    test_errors
)

best_lambda = lambdas[
    best_index
]

best_train_error = train_errors[
    best_index
]

best_test_error = test_errors[
    best_index
]


print("\n" + "=" * 60)
print("--- REGULARIZATION RESULTS ---")

print(
    f"Best alpha (lambda): "
    f"{best_lambda:.6f}"
)

print(
    f"Training MSE at best alpha: "
    f"{best_train_error:.6f}"
)

print(
    f"Testing MSE at best alpha: "
    f"{best_test_error:.6f}"
)


# 8. Plot Training and Testing Error Curves

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    lambdas,
    train_errors,
    label="Training Error"
)

plt.plot(
    lambdas,
    test_errors,
    label="Testing Error",
    linestyle="--"
)

plt.xscale(
    "log"
)

plt.xlabel(
    "Regularization Parameter (lambda / alpha)"
)

plt.ylabel(
    "Mean Squared Error"
)

plt.title(
    "Regularization Path: Ridge Regression "
    "Overfitting Control (Degree 15)"
)

plt.legend()

plt.grid(
    True,
    which="both",
    linestyle="--"
)

plt.tight_layout()

plt.show()