
import numpy as np

class DecisionTree:
    def __init__(
        self,
        max_depth=20,
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