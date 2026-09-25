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

Used when the relationship between variables is non-linear.

### Equation

Y = b₀ + b₁X + b₂X² + ... + bₙXⁿ

### Example

Predicting growth trends or market cycles.

## Polynomial Regression

Polynomial Regression is an extension of Linear Regression that models non-linear relationships between features and the target variable by introducing polynomial terms.

### Mathematical Formula

For degree 2:

\[
Y = b_0 + b_1X + b_2X^2
\]

For degree n:

\[
Y = b_0 + b_1X + b_2X^2 + ... + b_nX^n
\]

### Python Implementation

```python
import numpy as np


class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.coefficients = None

    def _create_polynomial_features(self, X):
        X = np.array(X).reshape(-1, 1)

        features = np.ones((len(X), 1))

        for d in range(1, self.degree + 1):
            features = np.hstack((features, X ** d))

        return features

    def fit(self, X, y):
        X_poly = self._create_polynomial_features(X)

        self.coefficients = np.linalg.inv(
            X_poly.T @ X_poly
        ) @ X_poly.T @ y

    def predict(self, X):
        X_poly = self._create_polynomial_features(X)
        return X_poly @ self.coefficients

    def score(self, X, y):
        predictions = self.predict(X)

        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - predictions) ** 2)

        return 1 - (ss_residual / ss_total)
```

### Example Usage

```python
import numpy as np

X = np.array([1, 2, 3, 4, 5, 6])
y = np.array([2, 5, 10, 17, 26, 37])

model = PolynomialRegression(degree=2)
model.fit(X, y)

predictions = model.predict(X)

print("Coefficients:", model.coefficients)
print("Predictions:", predictions)
print("R² Score:", model.score(X, y))
```

### Output

```text
Coefficients:
[1. 0. 1.]

Predictions:
[ 2.  5. 10. 17. 26. 37.]

R² Score:
1.0
```

The learned equation is:

\[
y = 1 + x^2
\]

### Gradient Descent Implementation

```python
import numpy as np


class PolynomialRegressionGD:
    def __init__(self, degree=2, learning_rate=0.001, epochs=5000):
        self.degree = degree
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    def _poly_features(self, X):
        X = np.array(X).reshape(-1, 1)

        features = np.ones((len(X), 1))

        for d in range(1, self.degree + 1):
            features = np.hstack((features, X ** d))

        return features

    def fit(self, X, y):
        X_poly = self._poly_features(X)

        samples, features = X_poly.shape

        self.weights = np.zeros(features)

        for _ in range(self.epochs):
            predictions = X_poly @ self.weights

            error = predictions - y

            gradient = (1 / samples) * (X_poly.T @ error)

            self.weights -= self.learning_rate * gradient

    def predict(self, X):
        X_poly = self._poly_features(X)
        return X_poly @ self.weights
```

### Scikit-Learn Implementation

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([2, 5, 10, 17, 26, 37])

model = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("linear", LinearRegression())
])

model.fit(X, y)

predictions = model.predict(X)

print(predictions)
```

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
