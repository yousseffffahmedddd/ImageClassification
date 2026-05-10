import numpy as np
import math
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# =========================
# DATA LOADING
# =========================
base_path = "C:/Users/sheha/Downloads/ImageClassification-main/dataset/"

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28 * 28) / 255.0

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)

def filter_binary_one_vs_all(X, y, positive_class):
    y_out = np.where(y == positive_class, 1, 0)
    return X, y_out


# =========================
# CUSTOM PCA
# =========================
class PCA_Custom:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]

        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        return np.dot(X - self.mean, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# =========================
# NAIVE BAYES
# =========================
class GaussianNB:
    def __init__(self):
        self.classes = []
        self.priors = {}
        self.means = {}
        self.vars = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples = len(X)

        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / n_samples
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0) + 1e-6

    def predict(self, X):
        y_pred = []

        for x in X:
            best_class = None
            best_score = -float('inf')

            for c in self.classes:
                log_probs = -0.5 * np.log(2 * np.pi * self.vars[c]) \
                            - ((x - self.means[c]) ** 2) / (2 * self.vars[c])

                score = np.log(self.priors[c]) + np.sum(log_probs)

                if score > best_score:
                    best_score = score
                    best_class = c

            y_pred.append(best_class)

        return np.array(y_pred)


# =========================
# EVALUATION
# =========================
def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }

def print_results(name, y_true, y_pred):
    print("=" * 60)
    print(name)
    print("=" * 60)

    print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred), 4))
    print("Recall   :", round(recall_score(y_true, y_pred), 4))
    print("F1-Score :", round(f1_score(y_true, y_pred), 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))


# =========================
# MANUAL K-FOLD SPLIT
# =========================
def manual_kfold_split(X, y, n_splits=5, shuffle=True, seed=42):
    np.random.seed(seed)

    indices = np.arange(len(X))

    if shuffle:
        np.random.shuffle(indices)

    fold_size = len(X) // n_splits
    folds = []

    for i in range(n_splits):
        start = i * fold_size
        end = (i + 1) * fold_size if i != n_splits - 1 else len(X)

        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        folds.append((train_idx, val_idx))

    return folds


# =========================
# MAIN
# =========================
def main():
    print("Loading data...")

    x_train = load_mnist_images(base_path + 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    y_train = load_mnist_labels(base_path + 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')

    x_test = load_mnist_images(base_path + 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    y_test = load_mnist_labels(base_path + 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    X_train_full, y_train_full = filter_binary_one_vs_all(x_train, y_train, 0)
    X_test, y_test = filter_binary_one_vs_all(x_test, y_test, 0)

    print("Train size:", len(X_train_full))
    print("Test size :", len(X_test))


    # =========================
    # MANUAL K-FOLD CV
    # =========================
    print("\nRunning Manual K-Fold Cross Validation...\n")

    folds = manual_kfold_split(X_train_full, y_train_full, n_splits=5, shuffle=True, seed=42)
    results = []

    for fold, (train_idx, val_idx) in enumerate(folds):

        X_train_fold = X_train_full[train_idx]
        y_train_fold = y_train_full[train_idx]

        X_val_fold = X_train_full[val_idx]
        y_val_fold = y_train_full[val_idx]

        # PCA per fold (NO leakage)
        pca = PCA_Custom(n_components=100)
        X_train_pca = pca.fit_transform(X_train_fold)
        X_val_pca = pca.transform(X_val_fold)

        model = GaussianNB()
        model.fit(X_train_pca, y_train_fold)

        y_pred = model.predict(X_val_pca)

        metrics = evaluate(y_val_fold, y_pred)
        results.append(metrics)

        print(f"Fold {fold+1} Accuracy:", round(metrics["accuracy"], 4))

    print("\nMean CV Accuracy:",
          round(np.mean([r["accuracy"] for r in results]), 4))


    # =========================
    # FINAL TRAINING
    # =========================
    print("\nTraining final model...")

    pca_final = PCA_Custom(n_components=100)
    X_train_full_pca = pca_final.fit_transform(X_train_full)
    X_test_pca = pca_final.transform(X_test)

    final_model = GaussianNB()
    final_model.fit(X_train_full_pca, y_train_full)

    y_test_pred = final_model.predict(X_test_pca)


    # =========================
    # FINAL RESULTS
    # =========================
    print_results("Final Test Results", y_test, y_test_pred)


if __name__ == "__main__":
    main()
