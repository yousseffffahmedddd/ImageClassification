import numpy as np

class DecisionTree:
    def __init__(
        self,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        n_features=100,
        n_thresholds=10,
        random_state=42
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_features = n_features
        self.n_thresholds = n_thresholds
        self.rng = np.random.default_rng(random_state)
        self.root = None

    # =========================
    # Train
    # =========================
    def fit(self, X, y):
        self.root = self._grow(X, y, depth=0)
        return self

    # =========================
    # Predict
    # =========================
    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    # =========================
    # Tree Growth
    # =========================
    def _grow(self, X, y, depth):

        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or np.all(y == y[0])
        ):
            return {
                "leaf": True,
                "value": np.mean(y)   # 🔥 probability instead of majority vote
            }

        n_samples, n_feats = X.shape

        feat_idxs = self.rng.choice(
            n_feats,
            min(self.n_features, n_feats),
            replace=False
        )

        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feat is None:
            return {"leaf": True, "value": np.mean(y)}

        left = X[:, best_feat] <= best_thresh
        right = ~left

        return {
            "leaf": False,
            "feature": best_feat,
            "threshold": best_thresh,
            "left": self._grow(X[left], y[left], depth + 1),
            "right": self._grow(X[right], y[right], depth + 1),
        }

    # =========================
    # Best Split (Gini)
    # =========================
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        parent_gini = self._gini(y)

        for feat in feat_idxs:
            X_col = X[:, feat]

            percentiles = np.linspace(5, 95, self.n_thresholds)
            thresholds = np.percentile(X_col, percentiles)

            for t in thresholds:
                left = X_col <= t
                right = ~left

                if (
                    left.sum() < self.min_samples_leaf
                    or right.sum() < self.min_samples_leaf
                ):
                    continue

                g_left = self._gini(y[left])
                g_right = self._gini(y[right])

                n = len(y)
                child = (left.sum() / n) * g_left + (right.sum() / n) * g_right

                gain = parent_gini - child

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat
                    split_thresh = t

        return split_idx, split_thresh

    # =========================
    # Gini
    # =========================
    def _gini(self, y):
        if len(y) == 0:
            return 0
        p = np.mean(y)
        return 1 - (p**2 + (1 - p)**2)

    # =========================
    # Leaf Decision (🔥 RECALIBRATED FOR RECALL)
    # =========================
    def _traverse(self, x, node):
        if node["leaf"]:
            # 🔥 LOWER threshold → MORE positives → higher recall
            return 1 if node["value"] >= 0.25 else 0

        if x[node["feature"]] <= node["threshold"]:
            return self._traverse(x, node["left"])
        else:
            return self._traverse(x, node["right"])

# =========================
# Flatten
# =========================
def extract_features(X):
    return X.reshape(len(X), -1)


# =========================
# PCA
# =========================
def apply_pca(X_train, X_test):
    pca = PCA(n_components=PCA_K)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca


# =========================
# Undersampling
# =========================
def balance_data(X, y):
    X_pos = X[y == 1]
    X_neg = X[y == 0]

    idx = np.random.choice(len(X_neg), len(X_pos), replace=False)
    X_neg_down = X_neg[idx]

    X_bal = np.vstack((X_neg_down, X_pos))
    y_bal = np.hstack((np.zeros(len(X_neg_down)), np.ones(len(X_pos))))

    perm = np.random.permutation(len(X_bal))
    return X_bal[perm], y_bal[perm]


# =========================
# K-Fold
# =========================
def k_fold_split(X, y, k=5):
    indices = np.random.permutation(len(X))
    folds = np.array_split(indices, k)

    for i in range(k):
        test_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])
        yield X[train_idx], y[train_idx], X[test_idx], y[test_idx]
""""
#(Main)
import os
import time
import numpy as np

from DT import DecisionTree
from sklearn.decomposition import PCA


# =========================
# Config
# =========================
DT_MAX_DEPTH = 14
DT_MIN_SAMPLES_LEAF = 3
PCA_K = 80   # 🔥 dimensionality reduction


# =========================
# Metrics
# =========================
def binary_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    return acc, prec, rec, f1, tn, fp, fn, tp


# =========================
# Print
# =========================
def print_results_table(title, metrics):
    acc, prec, rec, f1, tn, fp, fn, tp = metrics

    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(f"TN={tn}  FP={fp}")
    print(f"FN={fn}  TP={tp}")
    print("=" * 60)


# =========================
# Load Data
# =========================
def load_images(path):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)


def load_labels(path):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)


def get_dataset_path():
    base = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(base, "dataset"),
              os.path.join(os.path.dirname(base), "dataset")]:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError("Dataset not found")


# =========================
# Flatten
# =========================
def extract_features(X):
    return X.reshape(len(X), -1)


# =========================
# PCA
# =========================
def apply_pca(X_train, X_test):
    pca = PCA(n_components=PCA_K)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca


# =========================
# Undersampling
# =========================
def balance_data(X, y):
    X_pos = X[y == 1]
    X_neg = X[y == 0]

    idx = np.random.choice(len(X_neg), len(X_pos), replace=False)
    X_neg_down = X_neg[idx]

    X_bal = np.vstack((X_neg_down, X_pos))
    y_bal = np.hstack((np.zeros(len(X_neg_down)), np.ones(len(X_pos))))

    perm = np.random.permutation(len(X_bal))
    return X_bal[perm], y_bal[perm]


# =========================
# K-Fold
# =========================
def k_fold_split(X, y, k=5):
    indices = np.random.permutation(len(X))
    folds = np.array_split(indices, k)

    for i in range(k):
        test_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])
        yield X[train_idx], y[train_idx], X[test_idx], y[test_idx]


# =========================
# Train & Eval
# =========================
def train_and_eval(X_train, y_train, X_test, y_test, title):

    print(f"\n===== {title} =====")

    recalls = []

    for i, (X_tr, y_tr, X_val, y_val) in enumerate(k_fold_split(X_train, y_train, k=5)):

        model = DecisionTree(
            max_depth=DT_MAX_DEPTH,
            min_samples_leaf=DT_MIN_SAMPLES_LEAF
        )

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        tp = np.sum((y_val == 1) & (y_pred == 1))
        fn = np.sum((y_val == 1) & (y_pred == 0))
        rec = tp / (tp + fn)

        recalls.append(rec)

        print(f"Fold {i+1}: Recall = {rec:.4f}")

    print(f"Average Recall: {np.mean(recalls):.4f}")

    # Final model
    model = DecisionTree(
        max_depth=DT_MAX_DEPTH,
        min_samples_leaf=DT_MIN_SAMPLES_LEAF
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    print(f"Training time: {time.perf_counter() - t0:.2f}s")

    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    print(f"Prediction time: {time.perf_counter() - t1:.2f}s")

    print_results_table(title, binary_metrics(y_test, y_pred))


# =========================
# MAIN
# =========================
def main():

    base = get_dataset_path()

    x_train = load_images(os.path.join(base, "train-images-idx3-ubyte", "train-images-idx3-ubyte"))
    y_train = load_labels(os.path.join(base, "train-labels-idx1-ubyte", "train-labels-idx1-ubyte"))

    x_test = load_images(os.path.join(base, "t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte"))
    y_test = load_labels(os.path.join(base, "t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte"))

    # normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # binary task
    y_train = (y_train == 0).astype(int)
    y_test = (y_test == 0).astype(int)

    # flatten
    X_train = extract_features(x_train)
    X_test = extract_features(x_test)

    print(f"\nOriginal Shape: {X_train.shape}")

    # =========================
    # PCA APPLIED HERE
    # =========================
    X_train_pca, X_test_pca = apply_pca(X_train, X_test)

    print(f"PCA Shape: {X_train_pca.shape}")

    # =========================
    # NO BALANCING (PCA)
    # =========================
    train_and_eval(
        X_train_pca,
        y_train,
        X_test_pca,
        y_test,
        "RESULTS (PCA - NO BALANCING)"
    )

    # =========================
    # UNDERSAMPLING + PCA
    # =========================
    X_bal, y_bal = balance_data(X_train_pca, y_train)

    print(f"\nBalanced Shape: {X_bal.shape}")

    train_and_eval(
        X_bal,
        y_bal,
        X_test_pca,
        y_test,
        "RESULTS (PCA + UNDERSAMPLING)"
    )


if __name__ == "__main__":
    main()

"""