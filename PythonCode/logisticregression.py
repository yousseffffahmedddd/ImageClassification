import numpy as np
class LogisticRegression:
    def __init__(self, lr=0.5, epochs=300, reg_lambda=1.0):
        self.lr = lr
        self.epochs = epochs
        self.reg_lambda = reg_lambda

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y, y_pred):
        eps = 1e-8
        ce = -np.mean(y * np.log(y_pred + eps) +
                      (1 - y) * np.log(1 - y_pred + eps))
        l2 = (self.reg_lambda / (2 * len(y))) * np.sum(self.w ** 2)
        return ce + l2

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        self.losses = []

        for epoch in range(self.epochs):
            linear = np.dot(X, self.w) + self.b
            y_pred = self.sigmoid(linear)

            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y)) + (self.reg_lambda/n_samples) * self.w
            db = (1/n_samples) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 20 == 0:
                print(f"Epoch {epoch} | Loss: {loss:.4f}")

    def predict(self, X):
        probs = self.sigmoid(np.dot(X, self.w) + self.b)
        return (probs >= 0.5).astype(int)
