import numpy as np
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

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

# =========================
# FILTER BINARY (0 vs 1)
# =========================
def filter_binary(X, y, c1, c2):
    mask = (y == c1) | (y == c2)
    X_out = X[mask]
    y_out = np.where(y[mask] == c1, 0, 1) # Remap to 0 and 1
    return X_out, y_out

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
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)
        n_samples = len(X)

        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / n_samples
            
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0) + 1e-6   

    def predict(self, X):
        X = np.array(X)
        y_pred = []
        
        for x in X:
            best_class = None
            best_score = -float('inf')

            for c in self.classes:
                # Vectorized log probability calculation for the entire feature vector
                log_probs = -0.5 * np.log(2 * np.pi * self.vars[c]) - ((x - self.means[c]) ** 2) / (2 * self.vars[c])
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
# MAIN
# =========================
def main():
    print("Loading data...")

    x_train = load_mnist_images(base_path + 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    y_train = load_mnist_labels(base_path + 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')

    x_test = load_mnist_images(base_path + 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    y_test = load_mnist_labels(base_path + 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    
    X_train_full, y_train_full = filter_binary(x_train, y_train, 0, 1)
    X_test, y_test = filter_binary(x_test, y_test, 0, 1)

    print("Filtered Train:", len(X_train_full))
    print("Filtered Test :", len(X_test))

    # =========================
    # K-FOLD CROSS VALIDATION
    # =========================
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    print("\nRunning K-Fold Cross Validation...\n")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
        X_train_fold = X_train_full[train_idx]
        y_train_fold = y_train_full[train_idx]

        X_val_fold = X_train_full[val_idx]
        y_val_fold = y_train_full[val_idx]

        
        pca = PCA(n_components=100)
        X_train_pca = pca.fit_transform(X_train_fold)
        X_val_pca = pca.transform(X_val_fold)

        model = GaussianNB()
        model.fit(X_train_pca, y_train_fold)

        y_pred = model.predict(X_val_pca)

        metrics = evaluate(y_val_fold, y_pred)
        results.append(metrics)

        print(f"Fold {fold+1} Accuracy:", round(metrics["accuracy"], 4))

    # =========================
    # FINAL TRAINING ON FULL DATA
    # =========================
    print("\nTraining final model...")

    # Apply PCA to the entire training set for the final model
    pca_final = PCA(n_components=100)
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
