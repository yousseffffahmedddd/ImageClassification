import numpy as np
from scipy import stats

class KNN:
    def __init__(self, k=3, batch_size=500):
        self.k = k
        self.batch_size = batch_size

    def fit(self, X, y):
        self.X_train = X.astype(np.float32)
        self.y_train = y

    def predict(self, X):
        X = X.astype(np.float32)
        n_test = X.shape[0]
        predictions = np.empty(n_test, dtype=self.y_train.dtype)
        train_norms = np.sum(self.X_train**2, axis=1)

        for start in range(0, n_test, self.batch_size):
            end = min(start + self.batch_size, n_test)
            batch = X[start:end]

            dists = (
                np.sum(batch**2, axis=1, keepdims=True)
                + train_norms
                - 2 * (batch @ self.X_train.T)
            )

            k_indices = np.argpartition(dists, self.k - 1, axis=1)[:, :self.k]
            k_labels  = self.y_train[k_indices]
            predictions[start:end] = stats.mode(k_labels, axis=1).mode.flatten()
        return predictions