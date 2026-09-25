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

Ridge Regression is a regularized version of Linear Regression that adds an **L2 penalty** term to the cost function. This penalty discourages large coefficient values, helping to reduce overfitting and improve model generalization.

### Cost Function

J = RSS + λΣβ²

Advantages:
- Reduces overfitting
- Handles multicollinearity


### Mathematical Formula

### Cost Function

#### Linear Regression

Linear Regression minimizes the **Residual Sum of Squares (RSS)**:

```text
RSS = Σ(yi - ŷi)²
```

Where:

- `yi` = Actual value
- `ŷi` = Predicted value
- `n` = Number of observations

#### Ridge Regression

Ridge Regression adds an **L2 Regularization** penalty to the Linear Regression cost function:

```text
Cost = RSS + λ × Σ(βj²)
```

Where:

| Symbol | Description |
|---------|------------|
| RSS | Residual Sum of Squares |
| λ | Regularization Parameter |
| βj | Coefficient of Feature j |
| p | Total Number of Features |

The objective is to minimize both:

1. Prediction Error (RSS)
2. Magnitude of Coefficients (L2 Penalty)

As the value of **λ (Lambda)** increases, the coefficients shrink toward zero, helping to reduce overfitting and improve model generalization.

### Effect of λ (Lambda)

| Lambda Value | Effect |
|-------------|---------|
| λ = 0 | Equivalent to Linear Regression |
| Small λ | Slight regularization |
| Large λ | Strong regularization |
| Very Large λ | Underfitting |

## Why Use Ridge Regression?

- Reduces overfitting
- Handles multicollinearity
- Stabilizes coefficient estimates
- Improves generalization
- Works well with many correlated features

## Linear Regression vs Ridge Regression

| Feature | Linear Regression | Ridge Regression |
|----------|------------------|------------------|
| Regularization | ❌ | ✅ L2 |
| Handles Multicollinearity | ❌ | ✅ |
| Overfitting Control | ❌ | ✅ |
| Feature Selection | ❌ | ❌ |
| Coefficients Shrinkage | ❌ | ✅ |

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

## Advantages

- Prevents overfitting
- Handles correlated features
- Improves model stability
- Reduces coefficient variance
- Works well with high-dimensional data

## Limitations

- Does not perform feature selection
- All features remain in the model
- Requires tuning of alpha
- Less interpretable than sparse models

## Applications

- Stock price prediction
- Financial forecasting
- Healthcare analytics
- Demand forecasting
- Marketing analytics
- Risk assessment
- Economic modeling

## Best Practices

1. Standardize features before training.
2. Tune alpha using cross-validation.
3. Compare against Linear Regression baseline.
4. Use Ridge when multicollinearity exists.
5. Monitor validation performance to avoid underfitting.

## Conclusion

Ridge Regression extends Linear Regression by introducing L2 regularization. It reduces overfitting, improves model stability, and performs particularly well when features are highly correlated. It is one of the most widely used regularization techniques in machine learning and predictive analytics.

---

## 4. Lasso Regression

Lasso (Least Absolute Shrinkage and Selection Operator) Regression is a regularized version of Linear Regression that adds an **L1 penalty** to the cost function. Unlike Ridge Regression, Lasso can shrink some coefficients exactly to zero, effectively performing **feature selection**.

### Cost Function

J = RSS + λΣ|β|

Advantages:
- Feature selection
- Removes less important features

### Cost Function

#### Linear Regression

Linear Regression minimizes the **Residual Sum of Squares (RSS)**:

```text
RSS = Σ(yi - ŷi)²
```

#### Lasso Regression

Lasso Regression adds an **L1 Regularization** penalty to the Linear Regression cost function:

```text
Cost = RSS + λ × Σ|βj|
```

Where:

| Symbol | Description |
|---------|------------|
| RSS | Residual Sum of Squares |
| λ | Regularization Parameter |
| βj | Coefficient of Feature j |
| p | Total Number of Features |

The objective is to minimize both:

1. Prediction Error (RSS)
2. Sum of Absolute Coefficient Values (L1 Penalty)

As the value of **λ (Lambda)** increases, more coefficients are pushed toward zero. Some coefficients may become exactly zero, automatically removing unimportant features.

### Why Use Lasso Regression?

- Reduces overfitting
- Performs automatic feature selection
- Handles high-dimensional datasets
- Produces simpler and more interpretable models
- Removes irrelevant features

## Linear vs Ridge vs Lasso

| Feature | Linear | Ridge | Lasso |
|----------|---------|--------|--------|
| Regularization | ❌ | L2 | L1 |
| Feature Selection | ❌ | ❌ | ✅ |
| Handles Multicollinearity | ❌ | ✅ | ✅ |
| Coefficient Shrinkage | ❌ | ✅ | ✅ |
| Coefficients Become Zero | ❌ | ❌ | ✅ |
| Interpretability | Medium | Medium | High |

---

### Choosing Alpha (λ)

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

Grid Search:

```python
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

params = {
    "alpha": [0.001, 0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(
    Lasso(),
    params,
    cv=5
)

grid.fit(X, y)

print(grid.best_params_)
```


## Advantages

- Automatic feature selection
- Reduces overfitting
- Produces sparse models
- Improves interpretability
- Works well with high-dimensional data

## Limitations

- Can remove useful correlated features
- Requires feature scaling
- Sensitive to the alpha parameter
- Optimization is more complex than Linear Regression

## Applications

- Financial forecasting
- Stock market prediction
- Healthcare analytics
- Customer churn prediction
- Marketing analytics
- Risk modeling
- Feature selection in large datasets

## Best Practices

1. Standardize features before training.
2. Tune alpha using cross-validation.
3. Compare results with Ridge Regression.
4. Use when feature selection is important.
5. Monitor validation performance to avoid underfitting.

## Conclusion

Lasso Regression extends Linear Regression by adding L1 regularization. Its key advantage is automatic feature selection, making it especially useful for high-dimensional datasets where many features may be irrelevant. By shrinking some coefficients exactly to zero, Lasso produces simpler, more interpretable, and often more robust models.

---

# 5. Elastic Net Regression

Elastic Net Regression combines the strengths of both **Ridge Regression (L2 Regularization)** and **Lasso Regression (L1 Regularization)**. It is particularly useful when dealing with datasets that contain many correlated features.


## Cost Function

#### Linear Regression

Linear Regression minimizes the **Residual Sum of Squares (RSS)**:

```text
RSS = Σ(yi - ŷi)²
```

#### Elastic Net Regression

Elastic Net adds both **L1** and **L2** penalties to the cost function:

```text
Cost = RSS + λ1 × Σ|βj| + λ2 × Σ(βj²)
```

Where:

| Symbol | Description |
|---------|------------|
| RSS | Residual Sum of Squares |
| λ1 | L1 Regularization Parameter |
| λ2 | L2 Regularization Parameter |
| βj | Coefficient of Feature j |
| p | Total Number of Features |

The objective is to minimize:

1. Prediction Error (RSS)
2. L1 Penalty (Feature Selection)
3. L2 Penalty (Coefficient Shrinkage)

### Why Use Elastic Net?

- Combines Ridge and Lasso advantages
- Performs feature selection
- Handles multicollinearity
- Reduces overfitting
- Works well with high-dimensional datasets
- Better generalization


## Understanding l1_ratio

| l1_ratio | Behavior |
|-----------|----------|
| 0.0 | Pure Ridge Regression |
| 0.25 | Mostly Ridge |
| 0.50 | Balanced Ridge + Lasso |
| 0.75 | Mostly Lasso |
| 1.0 | Pure Lasso Regression |


## Linear vs Ridge vs Lasso vs Elastic Net

| Feature | Linear | Ridge | Lasso | Elastic Net |
|----------|---------|--------|--------|-------------|
| Regularization | ❌ | L2 | L1 | L1 + L2 |
| Feature Selection | ❌ | ❌ | ✅ | ✅ |
| Multicollinearity Handling | ❌ | ✅ | Partial | ✅ |
| Coefficient Shrinkage | ❌ | ✅ | ✅ | ✅ |
| Sparse Model | ❌ | ❌ | ✅ | ✅ |
| Overfitting Control | ❌ | ✅ | ✅ | ✅ |


## Hyperparameter Tuning

```python
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

params = {
    "alpha": [0.001, 0.01, 0.1, 1, 10],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
}

grid = GridSearchCV(
    ElasticNet(),
    params,
    cv=5
)

grid.fit(X, y)

print(grid.best_params_)
```

## Advantages

- Combines strengths of Ridge and Lasso
- Performs automatic feature selection
- Handles correlated features effectively
- Reduces overfitting
- Suitable for high-dimensional datasets

## Limitations

- Requires tuning two hyperparameters
- More computationally expensive
- Interpretation is slightly more complex
- Requires feature scaling

## Applications

- Stock Market Prediction
- Financial Forecasting
- Healthcare Analytics
- Marketing Analytics
- Customer Churn Prediction
- Demand Forecasting
- Risk Assessment
- High-Dimensional Machine Learning Problems

## Best Practices

1. Standardize features before training.
2. Use cross-validation for alpha and l1_ratio.
3. Compare against Ridge and Lasso baselines.
4. Monitor validation metrics to avoid underfitting.
5. Use Elastic Net when many features are correlated.


## Conclusion

Elastic Net Regression combines the feature selection capability of Lasso Regression with the coefficient stabilization of Ridge Regression. It is often the preferred choice for high-dimensional datasets containing correlated features, providing a balance between model simplicity, stability, and predictive performance.

---

# 6. Support Vector Regression (SVR)

Support Vector Regression (SVR) is a regression algorithm based on **Support Vector Machines (SVM)**. Unlike Linear Regression, SVR attempts to fit the best line (or hyperplane) within a specified error margin called **epsilon (ε)** while maximizing the margin around the prediction boundary.

SVR is highly effective for both **linear** and **non-linear** regression problems.

Advantages:
- Works well on non-linear data
- Robust to outliers

## Cost Function

#### Linear Regression

Linear Regression minimizes:

```text
RSS = Σ(yi - ŷi)²
```

#### Support Vector Regression

SVR minimizes:

```text
½ ||w||² + C × Σ(ξi + ξi*)
```

Subject to:

```text
|yi - ŷi| ≤ ε
```

Where:

| Symbol | Description |
|---------|------------|
| w | Weight Vector |
| C | Regularization Parameter |
| ε | Epsilon Margin |
| ξi | Slack Variable |
| ξi* | Slack Variable |
| yi | Actual Value |
| ŷi | Predicted Value |

The objective is:

1. Maximize Margin
2. Minimize Prediction Error
3. Control Model Complexity


### Why Use SVR?

- Handles non-linear relationships
- Works well on small and medium datasets
- Robust to outliers
- Effective in high-dimensional spaces
- Supports multiple kernel functions


## Types of SVR Kernels

#### 1. Linear Kernel

```text
K(xi, xj) = xi · xj
```

Used when data is approximately linear.

#### 2. Polynomial Kernel

```text
K(xi, xj) = (γ(xi · xj) + r)^d
```

Used for polynomial relationships.

#### 3. Radial Basis Function (RBF)

```text
K(xi, xj) = exp(-γ ||xi - xj||²)
```

Most commonly used kernel.

#### 4. Sigmoid Kernel

```text
K(xi, xj) = tanh(γ(xi · xj) + r)
```

Inspired by neural networks.


## Hyperparameters

#### 1. C (Regularization)

Controls trade-off between:

- Smooth Model
- Accurate Predictions

| C Value | Behavior |
|----------|-----------|
| Small C | More Regularization |
| Large C | Less Regularization |

#### 2. Epsilon (ε)

Defines acceptable prediction error.

| ε Value | Behavior |
|----------|-----------|
| Small ε | More Sensitive |
| Large ε | Less Sensitive |

#### 3. Gamma (γ)

Controls influence of training samples.

| Gamma | Behavior |
|---------|----------|
| Small | Smooth Boundary |
| Large | Complex Boundary |


## Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

params = {
    "C": [0.1, 1, 10, 100],
    "epsilon": [0.01, 0.1, 0.5, 1],
    "gamma": ["scale", "auto"],
    "kernel": ["rbf", "linear"]
}

grid = GridSearchCV(
    SVR(),
    params,
    cv=5
)

grid.fit(X, y)

print(grid.best_params_)
```


## Feature Scaling

SVR is highly sensitive to feature scaling.

Always standardize features before training:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(
        kernel="rbf",
        C=100,
        epsilon=0.1
    ))
])

model.fit(X_train, y_train)
```


## Linear Regression vs SVR

| Feature | Linear Regression | SVR |
|----------|------------------|-----|
| Linear Data | ✅ | ✅ |
| Non-Linear Data | ❌ | ✅ |
| Outlier Robustness | ❌ | ✅ |
| Kernel Support | ❌ | ✅ |
| Feature Scaling Required | Optional | Recommended |
| Computational Cost | Low | High |


## Advantages

- Handles non-linear data effectively
- Robust to outliers
- Supports multiple kernels
- Works well in high-dimensional spaces
- Strong generalization capability


## Limitations

- Computationally expensive on large datasets
- Requires feature scaling
- Hyperparameter tuning can be challenging
- Training time increases with dataset size


## Applications

- Stock Price Prediction
- Demand Forecasting
- Energy Consumption Prediction
- Time Series Regression
- Healthcare Analytics
- Economic Forecasting
- Risk Modeling
- Scientific Data Analysis


## Best Practices

1. Standardize all features.
2. Start with the RBF kernel.
3. Tune C, epsilon, and gamma using Grid Search.
4. Use cross-validation for evaluation.
5. Compare performance against Linear Regression and Random Forest Regression.


## Conclusion

Support Vector Regression (SVR) extends the principles of Support Vector Machines to regression tasks. By introducing an epsilon-insensitive margin and kernel-based learning, SVR can model complex non-linear relationships while maintaining strong generalization performance. It is particularly effective for medium-sized datasets where prediction accuracy is more important than training speed.

---

## 7. Decision Tree Regression

Decision Tree Regression is a non-linear supervised learning algorithm that predicts continuous values by recursively splitting the dataset into smaller subsets based on feature values. The model creates a tree-like structure where each internal node represents a decision rule and each leaf node contains the predicted value.

Unlike Linear Regression, Decision Tree Regression can capture complex non-linear relationships without requiring feature transformations.

### How Decision Trees Work

1. Start with the entire dataset.
2. Find the best feature and split point.
3. Divide the data into child nodes.
4. Repeat recursively for each child node.
5. Stop when a stopping criterion is met.
6. Use the average target value in a leaf node as the prediction.


### Splitting Criterion

Decision Tree Regression commonly uses **Mean Squared Error (MSE)** to determine the best split.

```text
MSE = (1 / n) × Σ(yi - ȳ)²
```

Where:

| Symbol | Description |
|---------|------------|
| yi | Actual Value |
| ȳ | Mean Value of Node |
| n | Number of Samples |

The algorithm selects the split that minimizes the overall MSE.

### Why Use Decision Tree Regression?

- Captures non-linear relationships
- No feature scaling required
- Easy to interpret
- Handles numerical and categorical data
- Works well with complex datasets


## Visualizing the Tree

```python
from sklearn.tree import (
    plot_tree
)
import matplotlib.pyplot as plt

plt.figure(
    figsize=(12, 8)
)

plot_tree(
    model,
    filled=True,
    rounded=True
)

plt.show()
```


## Important Hyperparameters

### max_depth

Maximum depth of the tree.

```python
DecisionTreeRegressor(
    max_depth=5
)
```

| Value | Effect |
|---------|---------|
| Small | Underfitting |
| Large | Overfitting |


### min_samples_split

Minimum samples required to split a node.

```python
DecisionTreeRegressor(
    min_samples_split=10
)
```


### min_samples_leaf

Minimum samples required in a leaf node.

```python
DecisionTreeRegressor(
    min_samples_leaf=5
)
```

### max_features

Number of features considered for splitting.

```python
DecisionTreeRegressor(
    max_features="sqrt"
)
```

## Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

params = {
    "max_depth": [3, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5]
}

grid = GridSearchCV(
    DecisionTreeRegressor(),
    params,
    cv=5
)

grid.fit(X, y)

print(grid.best_params_)
```

## Feature Importance

Decision Trees can estimate feature importance.

```python
for feature, importance in zip(
    X.columns,
    model.feature_importances_
):
    print(
        feature,
        importance
    )
```

Example Output:

```text
Age          0.42
Income       0.33
Experience   0.25
```

### Linear Regression vs Decision Tree Regression

| Feature | Linear Regression | Decision Tree |
|----------|------------------|--------------|
| Linear Data | ✅ | ✅ |
| Non-Linear Data | ❌ | ✅ |
| Feature Scaling Required | ❌ | ❌ |
| Interpretability | High | High |
| Handles Outliers | Poor | Better |
| Captures Complex Patterns | ❌ | ✅ |

### Advantages

- Handles non-linear relationships
- Easy to understand and visualize
- No feature scaling required
- Works with numerical and categorical data
- Captures interactions automatically

### Limitations

- Prone to overfitting
- Sensitive to small data changes
- Can create complex trees
- Lower generalization than ensemble methods

### Applications

- House Price Prediction
- Stock Market Analysis
- Demand Forecasting
- Customer Lifetime Value Prediction
- Sales Forecasting
- Risk Assessment
- Energy Consumption Prediction
- Healthcare Analytics

### Best Practices

1. Limit tree depth to avoid overfitting.
2. Use cross-validation for model selection.
3. Tune max_depth and min_samples_leaf.
4. Compare performance with Random Forest and Gradient Boosting.
5. Monitor training vs validation performance.

### Conclusion

Decision Tree Regression is a powerful non-linear machine learning algorithm that recursively partitions data to make predictions. It is easy to interpret, requires minimal preprocessing, and can capture complex relationships between variables. However, because individual trees are prone to overfitting, they are often used as the foundation for more advanced ensemble methods such as Random Forest and Gradient Boosting.


---

# 8. Random Forest Regression

Random Forest Regression is an ensemble learning algorithm that combines multiple Decision Trees to improve prediction accuracy and reduce overfitting. Instead of relying on a single decision tree, Random Forest builds many trees using different subsets of data and features, then averages their predictions.

It is one of the most popular machine learning algorithms for regression tasks because it can capture complex non-linear relationships while maintaining strong generalization performance.


## How Random Forest Works

1. Create multiple bootstrap samples from the training dataset.
2. Build a Decision Tree for each sample.
3. At each split, consider only a random subset of features.
4. Train all trees independently.
5. Average predictions from all trees.

```text
Training Data
       ↓
Bootstrap Sampling
       ↓
Decision Tree 1
Decision Tree 2
Decision Tree 3
...
Decision Tree N
       ↓
Average Predictions
       ↓
Final Prediction
```


## Prediction Formula

For a Random Forest containing N trees:

```text
Prediction = (Tree1 + Tree2 + Tree3 + ... + TreeN) / N
```

or

```text
ŷ = (1/N) × Σ(Treei)
```

Where:

| Symbol | Description |
|---------|------------|
| ŷ | Final Prediction |
| N | Number of Trees |
| Treei | Prediction from Tree i |


## Why Use Random Forest Regression?

- Handles non-linear relationships
- Reduces overfitting
- High prediction accuracy
- Robust to noise and outliers
- Automatically captures feature interactions
- Works with large datasets


## Important Hyperparameters

### 1. n_estimators

Number of trees in the forest.

```python
RandomForestRegressor(
    n_estimators=100
)
```

| Value | Effect |
|---------|---------|
| Small | Faster Training |
| Large | Better Accuracy |


### 2. max_depth

Maximum depth of each tree.

```python
RandomForestRegressor(
    max_depth=10
)
```

| Value | Effect |
|---------|---------|
| Small | Underfitting |
| Large | Overfitting Risk |


### 3. min_samples_split

Minimum samples required to split a node.

```python
RandomForestRegressor(
    min_samples_split=5
)
```


### 4. min_samples_leaf

Minimum samples required in a leaf node.

```python
RandomForestRegressor(
    min_samples_leaf=2
)
```


### 5. max_features

Number of features considered for each split.

```python
RandomForestRegressor(
    max_features="sqrt"
)
```


## Hyperparameter Tuning

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5]
}

grid = GridSearchCV(
    RandomForestRegressor(),
    params,
    cv=5
)

grid.fit(X, y)

print(grid.best_params_)
```

## Decision Tree vs Random Forest

| Feature | Decision Tree | Random Forest |
|----------|--------------|--------------|
| Accuracy | Medium | High |
| Overfitting | High | Low |
| Stability | Low | High |
| Training Speed | Fast | Moderate |
| Interpretability | High | Medium |
| Generalization | Medium | High |


## Advantages

- High predictive accuracy
- Handles non-linear relationships
- Resistant to overfitting
- Robust to outliers and noise
- Supports feature importance analysis
- Minimal preprocessing required


## Limitations

- Slower than a single Decision Tree
- Higher memory consumption
- Less interpretable
- Large models can be computationally expensive


## Applications

- Stock Price Prediction
- Financial Forecasting
- House Price Prediction
- Demand Forecasting
- Customer Lifetime Value Prediction
- Healthcare Analytics
- Risk Assessment
- Energy Consumption Forecasting


## Best Practices

1. Start with 100–200 trees.
2. Use cross-validation for hyperparameter tuning.
3. Analyze feature importance.
4. Limit tree depth if overfitting occurs.
5. Compare against Gradient Boosting models.


## Time Complexity

Let:

```text
N = Number of Samples
M = Number of Features
T = Number of Trees
```

Training Complexity:

```text
O(T × N × M × log(N))
```

Prediction Complexity:

```text
O(T × log(N))
```


## Conclusion

Random Forest Regression is an ensemble machine learning algorithm that combines multiple Decision Trees to produce highly accurate and stable predictions. By averaging predictions from many trees, it significantly reduces overfitting while maintaining the ability to model complex non-linear relationships. It is widely used in finance, healthcare, forecasting, and predictive analytics due to its robustness and strong performance on real-world datasets.

---

# 9. Gradient Boosting Regression

Gradient Boosting Regression is an ensemble machine learning algorithm that builds a sequence of Decision Trees, where each new tree attempts to correct the errors made by the previous trees.

Unlike Random Forest, which builds trees independently and averages their predictions, Gradient Boosting builds trees sequentially and combines them to minimize the overall prediction error.

It is one of the most powerful predictive modeling techniques and forms the foundation of popular algorithms such as:

- XGBoost
- LightGBM
- CatBoost


## How Gradient Boosting Works

1. Train the first Decision Tree.
2. Calculate prediction errors (residuals).
3. Train a new tree on the residuals.
4. Add the new tree's predictions to the previous model.
5. Repeat until the desired number of trees is reached.

```text
Actual Value
      ↓
Tree 1
      ↓
Residual Error
      ↓
Tree 2
      ↓
Residual Error
      ↓
Tree 3
      ↓
...
      ↓
Final Prediction
```

Each new tree focuses on correcting mistakes made by earlier trees.


## Prediction Formula

For M trees:

```text
Prediction = Tree1 + LearningRate × Tree2
           + LearningRate × Tree3
           + ...
           + LearningRate × TreeM
```

or

```text
ŷ = Σ(η × Treei)
```

Where:

| Symbol | Description |
|---------|------------|
| ŷ | Final Prediction |
| η | Learning Rate |
| Treei | Prediction from Tree i |
| M | Number of Trees |


## Why Use Gradient Boosting?

- High predictive accuracy
- Handles non-linear relationships
- Captures complex feature interactions
- Reduces bias and variance
- Works well on structured/tabular data
- State-of-the-art performance on many datasets


## Important Hyperparameters

### n_estimators

Number of boosting stages.

```python
GradientBoostingRegressor(
    n_estimators=100
)
```

| Value | Effect |
|---------|---------|
| Small | Faster Training |
| Large | Better Learning |
| Too Large | Overfitting Risk |


### learning_rate

Controls contribution of each tree.

```python
GradientBoostingRegressor(
    learning_rate=0.1
)
```

| Learning Rate | Effect |
|---------------|---------|
| High | Faster Learning |
| Low | Better Generalization |
| Very Low | Requires More Trees |


### max_depth

Maximum depth of each tree.

```python
GradientBoostingRegressor(
    max_depth=3
)
```

| Depth | Effect |
|--------|--------|
| Small | Simpler Model |
| Large | More Complex Model |


### subsample

Fraction of training samples used per tree.

```python
GradientBoostingRegressor(
    subsample=0.8
)
```

Values less than 1.0 introduce randomness and can improve generalization.


## Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

params = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0]
}

grid = GridSearchCV(
    GradientBoostingRegressor(),
    params,
    cv=5
)

grid.fit(X, y)

print(grid.best_params_)
```

--------------------------------------------------

## Random Forest vs Gradient Boosting

| Feature | Random Forest | Gradient Boosting |
|----------|--------------|------------------|
| Training Style | Parallel | Sequential |
| Overfitting Risk | Low | Medium |
| Accuracy | High | Very High |
| Training Speed | Faster | Slower |
| Hyperparameter Sensitivity | Low | High |
| Bias Reduction | Moderate | Excellent |

--------------------------------------------------

## Popular Gradient Boosting Variants

### XGBoost

Features:

- Regularization
- Parallel Processing
- Missing Value Handling
- High Performance

```python
from xgboost import XGBRegressor
```

--------------------------------------------------

### LightGBM

Features:

- Faster Training
- Lower Memory Usage
- Efficient for Large Datasets

```python
from lightgbm import LGBMRegressor
```

--------------------------------------------------

### CatBoost

Features:

- Native Categorical Feature Support
- Minimal Preprocessing
- Strong Default Performance

```python
from catboost import CatBoostRegressor
```

--------------------------------------------------

## Advantages

- Excellent prediction accuracy
- Captures complex patterns
- Handles non-linear relationships
- Supports feature importance analysis
- Strong performance on structured data

--------------------------------------------------

## Limitations

- Slower training
- More sensitive to hyperparameters
- Can overfit if not tuned properly
- Harder to interpret than a single tree

--------------------------------------------------

## Applications

- Stock Price Prediction
- Financial Forecasting
- Demand Forecasting
- House Price Prediction
- Customer Lifetime Value Prediction
- Fraud Detection
- Risk Assessment
- Healthcare Analytics

--------------------------------------------------

## Best Practices

1. Start with learning_rate = 0.1.
2. Use 100–500 trees initially.
3. Tune max_depth carefully.
4. Apply cross-validation.
5. Monitor training and validation scores.
6. Compare against XGBoost, LightGBM, and CatBoost.

--------------------------------------------------

## Time Complexity

Let:

```text
N = Number of Samples
M = Number of Features
T = Number of Trees
```

Training Complexity:

```text
O(T × N × M × log(N))
```

Prediction Complexity:

```text
O(T × log(N))
```

--------------------------------------------------

## Conclusion

Gradient Boosting Regression is a powerful ensemble learning algorithm that builds trees sequentially to correct previous prediction errors. By combining many weak learners into a strong predictive model, it achieves exceptional accuracy and is widely used in finance, forecasting, healthcare, and machine learning competitions. Modern implementations such as XGBoost, LightGBM, and CatBoost have made Gradient Boosting one of the most important algorithms in practical machine learning.


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
