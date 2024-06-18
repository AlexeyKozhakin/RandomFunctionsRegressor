import numpy as np
from sklearn.linear_model import Ridge
from scipy.linalg import lstsq
class RandomFunctionsRegressor:
  def __init__(self, C=1, n=1e5, t=1e5, method_reg = 'mnk'):
        self.C=C
        self.n=n
        self.t=t
        self.method_reg = method_reg

  def Kf(self, tau2):
    res = np.where(tau2 == 0, self.t, self.C * (tau2 * np.log(tau2 / self.t) + self.n))
    return res
  def abs_tau2(self, x, y):
      # Векторизованное вычисление квадрата расстояния между точками
      return np.sum((x - y)**2, axis=-1)


  def generate_matrix_for_linear_system(self, X_train, y_train):
      N = X_train.shape[0]
      A = np.zeros((N, N))

      # Используйте матричные операции для извлечения x и y из X_train
      X_tile = np.tile(X_train, (N, 1, 1))
      X_tile_transpose = np.transpose(X_tile, axes=(1, 0, 2))

      # Расчет матрицы A с использованием матричных операций
      abs_tau2_matrix = self.abs_tau2(X_tile, X_tile_transpose)
      A = self.Kf(abs_tau2_matrix)
      return A

  def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        b=y_train
        A = self.generate_matrix_for_linear_system(X_train, y_train)
        if self.method_reg=='mnk':
          # Решение с использованием МНК
          self.q, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        elif self.method_reg=='ridge':
          # Решение с использованием регрессии Ridge
          ridge = Ridge(alpha=0.1)  # Выберите подходящее значение alpha
          self.q = ridge.fit(A, b)
        elif self.method_reg == 'svd':
          U, S, Vt = np.linalg.svd(A)
          pseudoinv = np.dot(Vt.T, np.dot(np.diag(1/S), U.T))
          self.q = np.dot(pseudoinv, b)
        elif self.method_reg == 'lstsq':
          self.q, residuals, rank, s = lstsq(A, b)

  def predict(self, X):
    N = self.X_train.shape[0]
    K = X.shape[0]
    M = self.y_train.shape[1]
    y_pred = np.zeros((K, M))

    # Используйте матричные операции для извлечения x из X
    X_tile = np.tile(X, (N, 1, 1))
    X_train_tile = np.tile(self.X_train[:, np.newaxis, :], (1, K, 1))

    # Вычисление ядра для всех пар точек с использованием матричных операций
    abs_tau2_matrix = np.sum((X_tile - X_train_tile)**2, axis=-1)
    kernel_matrix = self.Kf(abs_tau2_matrix)

    # Перемножение ядра и коэффициентов для предсказания
    y_pred = np.dot(kernel_matrix.T, self.q)

    return y_pred


