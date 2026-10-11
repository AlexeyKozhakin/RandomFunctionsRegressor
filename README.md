# RandomFunctionsRegressor

RandomFunctionsRegressor is a Python library for random function regression.

For the Numerai experiments, reproduction on another machine, and the pending
Colab GPU work, start with [the agent handoff](experiments/HANDOFF.md) and
[experiment instructions](experiments/README.md). Raw datasets and fitted models
are intentionally excluded from Git; a verified downloader is included.

## LHS support-point regression

The new mode uses a fixed number of Latin hypercube support points instead of
one coefficient per training row. The original `Kf` function is unchanged:

\[
\hat y(x) = b + \sum_{j=1}^{m} a_j K_f(\|x-z_j\|^2).
\]

```python
from RandomFunctionsRegressor import RandomFunctionsRegressor

model = RandomFunctionsRegressor(
    n_support=128,
    random_state=42,
    method_reg="ridge",
    alpha=0.1,
    batch_size=1024,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(model.support_points_.shape)  # (128, number_of_input_features)
print(model.q.shape)                # (128,) for one target
```

`n_support` is the **total** number of support points, not the number raised to
the power of the feature count. Bounds are the minimum and maximum of each
training feature. Each varying coordinate has one point in each of the
`n_support` equal-width intervals; constant coordinates remain constant.
Prediction data do not change these bounds or support points. No input scaling
or missing-value imputation is performed: apply any preprocessing fitted on
training data before calling the estimator.

The training design matrix has shape `(training_rows, n_support)`. It is still
stored densely for the linear solve, so fit memory grows with both dimensions.
Distances and predictions are evaluated in batches, without three-dimensional
pairwise arrays. The fitted LHS model retains only support points, coefficients,
and training bounds, rather than the full training set.

- `method_reg="ridge"` minimizes squared error with coefficient penalty
  `alpha + sig**2`, using an SVD solver.
- `"mnk"`, `"lstsq"`, and `"svd"` solve rectangular least squares. Their
  coefficient penalty is `sig**2`; `alpha` is not used by these methods.
- `fit_intercept=True` adds a separate, unpenalized constant term.
- `n_support=None` (the default) preserves the original training-point mode.
  In that mode, the original meaning of `sig` as a matrix diagonal term remains.

The kernel parameter `n` is distinct from the new `n_support` parameter.
LHS stratifies individual coordinates; it does not guarantee uniform coverage
of every region of a high-dimensional space.

Run the LHS regression checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Installation

You can install the library directly from GitHub:

```
pip install git+https://github.com/AlexeyKozhakin/RandomFunctionsRegressor.git
```
Simple example
```

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from RandomFunctionsRegressor.core import RandomFunctionsRegressor

# Загружаем встроенный набор данных
data = load_diabetes()
X, y = data.data, data.target

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели RandomForest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Создание и обучение модели RandomFunctions
model_rfun = RandomFunctionsRegressor(method_reg='mnk',sig=0.25)
model_rfun.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)
# Предсказание на тестовой выборке
y_pred_rfun = model_rfun.predict(X_test)

# Вычисление RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Random Forest Root Mean Squared Error (RMSE): {rmse:.2f}')

rmse = np.sqrt(mean_squared_error(y_test, y_pred_rfun))
print(f'Random Functions Root Mean Squared Error (RMSE): {rmse:.2f}')

# График рассеивания фактических и предсказанных значений
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, edgecolor='k', alpha=0.7)
plt.scatter(y_test, y_pred_rfun, color='red', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.show()
```
