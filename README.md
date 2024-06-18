# RandomFunctionsRegressor

RandomFunctionsRegressor is a Python library for random function regression.

## Installation

You can install the library directly from GitHub:

```bash
pip install git+https://github.com/AlexeyKozhakin/RandomFunctionsRegressor.git

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
