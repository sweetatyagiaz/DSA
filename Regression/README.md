# Regression in Machine Learning

## Overview

Regression is a supervised machine learning technique used to predict continuous numerical values. It models the relationship between one or more independent variables (features) and a dependent variable (target).

Examples:
- Predicting house prices
- Forecasting stock prices
- Estimating sales revenue
- Predicting temperature

---

# What is Regression?

Regression attempts to find a mathematical relationship between input variables (X) and output variables (Y).

Mathematically:

Y = f(X) + ε

Where:
- Y = Dependent Variable (Target)
- X = Independent Variable (Feature)
- ε = Error Term

The goal is to minimize prediction error and find the best-fitting function.

---

# Types of Regression

## 1. Linear Regression

Models a linear relationship between variables.

### Equation

For Single Variable:

Y = b₀ + b₁X

Where:
- b₀ = Intercept
- b₁ = Slope

For Multiple Variables:

Y = b₀ + b₁X₁ + b₂X₂ + ... + bₙXₙ

### Example

Predict house price based on area.

| Area (sq ft) | Price ($) |
|--------------|-----------|
| 1000 | 150000 |
| 1500 | 200000 |
| 2000 | 250000 |

---

## 2. Polynomial Regression

Polynomial Regression is an extension of Linear Regression that models non-linear relationships between features and the target variable by introducing polynomial terms.

### Equation

Y = b₀ + b₁X + b₂X² + ... + bₙXⁿ

### Example

Predicting growth trends or market cycles.

### Time Complexity

| Operation | Complexity |
|------------|------------|
| Feature Generation | O(n × d) |
| Normal Equation Training | O(d³) |
| Prediction | O(n × d) |
| Gradient Descent Training | O(epoch × n × d) |

Where:

- `n` = Number of samples
- `d` = Polynomial degree

### Advantages

- Captures non-linear relationships
- Easy to implement and interpret
- Works well when the relationship is curved
- Compatible with standard linear regression solvers

### Limitations

- Can overfit for high polynomial degrees
- Sensitive to outliers
- Computational cost increases with degree
- Poor extrapolation outside training range

### When to Use Polynomial Regression

✅ Non-linear relationships

✅ Growth curves

✅ Demand forecasting

✅ Financial trend analysis

✅ Scientific modeling

✅ Sales and revenue forecasting

✅ Engineering and physical system modeling

### Best Practices

1. Start with degree 2 or 3.
2. Use cross-validation to select the optimal degree.
3. Scale features before training.
4. Monitor overfitting using validation data.
5. Consider Ridge or Lasso regularization for higher-degree models.

### Conclusion

Polynomial Regression extends Linear Regression by introducing polynomial features, allowing it to model curved and non-linear relationships. It is widely used in forecasting, scientific analysis, financial modeling, and engineering applications where simple linear relationships are insufficient.

---

## 3. Ridge Regression

Adds L2 Regularization to prevent overfitting.

### Cost Function

J = RSS + λΣβ²

Advantages:
- Reduces overfitting
- Handles multicollinearity


## Ridge Regression

Ridge Regression is a regularized version of Linear Regression that adds an **L2 penalty** term to the cost function. This penalty discourages large coefficient values, helping to reduce overfitting and improve model generalization.

### Mathematical Formula

Linear Regression minimizes:

\[
RSS = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
\]

Ridge Regression adds an L2 penalty:

\[
Cost = RSS + \lambda \sum_{j=1}^{p} \beta_j^2
\]

Where:

- \(RSS\) = Residual Sum of Squares
- \(\lambda\) = Regularization parameter
- \(\beta_j\) = Model coefficients

### Effect of λ (Lambda)

| Lambda Value | Effect |
|-------------|---------|
| λ = 0 | Equivalent to Linear Regression |
| Small λ | Slight regularization |
| Large λ | Strong regularization |
| Very Large λ | Underfitting |

### Why Use Ridge Regression?

- Reduces overfitting
- Handles multicollinearity
- Stabilizes coefficient estimates
- Improves generalization
- Works well with many correlated features

---

## Ridge Regression from Scratch

```python
import numpy as np


class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coefficients = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        rows, cols = X.shape

        identity = np.eye(cols)

        self.coefficients = np.linalg.inv(
            X.T @ X + self.alpha * identity
        ) @ X.T @ y

    def predict(self, X):
        X = np.array(X)
        return X @ self.coefficients

    def score(self, X, y):
        predictions = self.predict(X)

        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - predictions) ** 2)

        return 1 - (ss_residual / ss_total)
```

---

## Example Usage

```python
import numpy as np

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])

y = np.array([3, 5, 7, 9, 11])

model = RidgeRegression(alpha=1.0)

model.fit(X, y)

predictions = model.predict(X)

print("Coefficients:")
print(model.coefficients)

print("\nPredictions:")
print(predictions)

print("\nR² Score:")
print(model.score(X, y))
```

---

## Ridge Regression with Gradient Descent

```python
import numpy as np


class RidgeRegressionGD:
    def __init__(
        self,
        learning_rate=0.01,
        epochs=1000,
        alpha=1.0
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha

        self.weights = None
        self.bias = 0

    def fit(self, X, y):

        samples, features = X.shape

        self.weights = np.zeros(features)

        for _ in range(self.epochs):

            predictions = (
                np.dot(X, self.weights)
                + self.bias
            )

            error = predictions - y

            dw = (
                (1 / samples)
                * (X.T @ error)
                + (self.alpha / samples)
                * self.weights
            )

            db = (1 / samples) * np.sum(error)

            self.weights -= (
                self.learning_rate * dw
            )

            self.bias -= (
                self.learning_rate * db
            )

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
```

---

## Scikit-Learn Implementation

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

# Load data
data = pd.read_csv("data.csv")

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Ridge(alpha=1.0)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("R² Score:", r2_score(y_test, predictions))
```

---

## Linear Regression vs Ridge Regression

| Feature | Linear Regression | Ridge Regression |
|----------|------------------|------------------|
| Regularization | ❌ | ✅ L2 |
| Handles Multicollinearity | ❌ | ✅ |
| Overfitting Control | ❌ | ✅ |
| Feature Selection | ❌ | ❌ |
| Coefficients Shrinkage | ❌ | ✅ |

---

## Choosing the Alpha Parameter

Common values:

```python
alphas = [
    0.001,
    0.01,
    0.1,
    1,
    10,
    100
]
```

Use Grid Search:

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

params = {
    "alpha": [0.001, 0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(
    Ridge(),
    params,
    cv=5
)

grid.fit(X, y)

print(grid.best_params_)
```

---

## Advantages

- Prevents overfitting
- Handles correlated features
- Improves model stability
- Reduces coefficient variance
- Works well with high-dimensional data

---

## Limitations

- Does not perform feature selection
- All features remain in the model
- Requires tuning of alpha
- Less interpretable than sparse models

---

## Applications

- Stock price prediction
- Financial forecasting
- Healthcare analytics
- Demand forecasting
- Marketing analytics
- Risk assessment
- Economic modeling

---

## Best Practices

1. Standardize features before training.
2. Tune alpha using cross-validation.
3. Compare against Linear Regression baseline.
4. Use Ridge when multicollinearity exists.
5. Monitor validation performance to avoid underfitting.

---

## Conclusion

Ridge Regression extends Linear Regression by introducing L2 regularization. It reduces overfitting, improves model stability, and performs particularly well when features are highly correlated. It is one of the most widely used regularization techniques in machine learning and predictive analytics.

---

## 4. Lasso Regression

Adds L1 Regularization.

### Cost Function

J = RSS + λΣ|β|

Advantages:
- Feature selection
- Removes less important features

---

## 5. Elastic Net Regression

Combination of Ridge and Lasso.

### Cost Function

J = RSS + λ₁Σ|β| + λ₂Σβ²

Advantages:
- Feature selection
- Better generalization

---

## 6. Support Vector Regression (SVR)

Uses Support Vector Machine concepts for regression.

Advantages:
- Works well on non-linear data
- Robust to outliers

---

## 7. Decision Tree Regression

Uses tree-based splitting to predict continuous values.

Advantages:
- Easy interpretation
- Handles non-linear data

---

## 8. Random Forest Regression

Ensemble of multiple decision trees.

Advantages:
- High accuracy
- Reduces overfitting

---

## 9. Gradient Boosting Regression

Builds trees sequentially to reduce errors.

Popular implementations:
- XGBoost
- LightGBM
- CatBoost

Advantages:
- Excellent performance
- Handles complex relationships

---

# Linear Regression Visualization

The objective is to find the best-fit line that minimizes the distance between actual and predicted values.


::contentReference[oaicite:0]{index=0}


---

# Assumptions of Linear Regression

## 1. Linearity

Independent variables should have a linear relationship with the target.

## 2. Independence

Observations should be independent.

## 3. Homoscedasticity

Variance of residuals should remain constant.

## 4. Normality

Residuals should follow a normal distribution.

## 5. No Multicollinearity

Independent variables should not be highly correlated.

---

# Regression Workflow

```text
Data Collection
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Train-Test Split
       ↓
Model Training
       ↓
Prediction
       ↓
Evaluation
       ↓
Deployment
```

---

# Evaluation Metrics

## Mean Absolute Error (MAE)

MAE = (1/n) Σ |Yi - Ŷi|

Measures average absolute error.

Lower is better.

---

## Mean Squared Error (MSE)

MSE = (1/n) Σ (Yi - Ŷi)²

Penalizes large errors.

Lower is better.

---

## Root Mean Squared Error (RMSE)

RMSE = √MSE

Most commonly used metric.

Lower is better.

---

## R² Score

R² = 1 - (SSres / SStot)

Range:
- 1 → Perfect prediction
- 0 → No explanatory power
- Negative → Worse than baseline

Higher is better.


# Advantages of Regression

✅ Easy to understand

✅ Fast training

✅ Interpretable results

✅ Works well on structured data

✅ Useful for forecasting

---

# Limitations of Regression

❌ Sensitive to outliers

❌ Assumes relationships between variables

❌ Can overfit on noisy data

❌ Requires feature engineering

❌ Linear regression struggles with complex patterns

---

# Real-World Applications

## Finance
- Stock price prediction
- Risk modeling
- Revenue forecasting

## Healthcare
- Disease progression prediction
- Medical cost estimation

## Retail
- Demand forecasting
- Customer spending prediction

## Manufacturing
- Predictive maintenance
- Quality estimation

## Real Estate
- House price prediction
- Rental forecasting

---

# Comparison of Regression Algorithms

| Algorithm | Linear | Non-Linear | Feature Selection | Overfitting Control |
|------------|---------|------------|-------------------|--------------------|
| Linear Regression | ✅ | ❌ | ❌ | ❌ |
| Ridge | ✅ | ❌ | ❌ | ✅ |
| Lasso | ✅ | ❌ | ✅ | ✅ |
| Elastic Net | ✅ | ❌ | ✅ | ✅ |
| SVR | ✅ | ✅ | ❌ | ✅ |
| Decision Tree | ❌ | ✅ | ❌ | Moderate |
| Random Forest | ❌ | ✅ | ❌ | Good |
| XGBoost | ❌ | ✅ | Partial | Excellent |

---

# Best Practices

1. Handle missing values.
2. Remove duplicates.
3. Detect and treat outliers.
4. Scale features when required.
5. Perform feature engineering.
6. Use cross-validation.
7. Tune hyperparameters.
8. Monitor overfitting.
9. Select proper evaluation metrics.
10. Deploy and monitor model performance.

---

# Conclusion

Regression is one of the most fundamental machine learning techniques for predicting continuous values. From simple Linear Regression to advanced Gradient Boosting methods, regression models are widely used across finance, healthcare, manufacturing, retail, and many other domains. Selecting the right regression algorithm depends on the data characteristics, complexity of relationships, and business requirements.
