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

---

# Example Using Scikit-Learn

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Load data
data = pd.read_csv("house_prices.csv")

X = data[["area"]]
y = data["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
rmse = mean_squared_error(y_test, predictions, squared=False)
r2 = r2_score(y_test, predictions)

print("RMSE:", rmse)
print("R² Score:", r2)
```

---

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
