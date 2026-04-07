import numpy as np
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def predict(self, X):
        predictions = []

        for x in X:
            distances = [self.euclidean_distance(x, x_train)
                         for x_train in self.X_train]

            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]

            pred = np.bincount(k_labels).argmax()
            predictions.append(pred)

        return np.array(predictions)