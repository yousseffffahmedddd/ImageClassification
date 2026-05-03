import numpy as np
from scipy import stats
import numpy as np


# ── Basic Preprocessing ───────────────────────────────────────────────────────

def flatten_images(images):
    return images.reshape(images.shape[0], -1)


def normalize_images(images):
    return images / 255.0


# ── Class Imbalance ───────────────────────────────────────────────────────────

def random_oversample(X, y, random_state=42):
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_res, y_res = [X], [y]

    for cls, count in zip(classes, counts):
        if count < max_count:
            idx = np.where(y == cls)[0]
            extra = rng.choice(idx, size=max_count - count, replace=True)
            X_res.append(X[extra])
            y_res.append(y[extra])

    X_out = np.vstack(X_res)
    y_out = np.concatenate(y_res)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]


def random_undersample(X, y, random_state=42):
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    X_res, y_res = [], []

    for cls in classes:
        idx = np.where(y == cls)[0]
        chosen = rng.choice(idx, size=min_count, replace=False)
        X_res.append(X[chosen])
        y_res.append(y[chosen])

    X_out = np.vstack(X_res)
    y_out = np.concatenate(y_res)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]


# ── PCA ───────────────────────────────────────────────────────────────────────

def pca(X_train, X_test, n_components=50):
    mean_vector      = np.mean(X_train, axis=0)
    X_train_centered = X_train - mean_vector
    X_test_centered  = X_test  - mean_vector

    cov_matrix                = np.cov(X_train_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_idx   = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]
    components   = eigenvectors[:, :n_components]

    X_train_pca = X_train_centered @ components
    X_test_pca  = X_test_centered  @ components

    return X_train_pca, X_test_pca, components


# ── HOG ───────────────────────────────────────────────────────────────────────

def hog_features(images, cell_size=4, block_size=2, bins=9):
    # Handle flat (n, 784) or shaped (n, 28, 28) input
    if images.ndim == 2:
        n    = images.shape[0]
        side = int(np.sqrt(images.shape[1]))
        images = images.reshape(n, side, side)

    n_samples, h, w = images.shape
    eps = 1e-6
    hog_all = []

    for img in images:
        img = img.astype(np.float32)

        gx = np.zeros_like(img)
        gy = np.zeros_like(img)
        gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
        gy[1:-1, :] = img[2:, :] - img[:-2, :]

        magnitude   = np.sqrt(gx**2 + gy**2)
        orientation = (np.arctan2(gy, gx) * 180 / np.pi) % 180

        n_cells_x   = w // cell_size
        n_cells_y   = h // cell_size
        hist_tensor = np.zeros((n_cells_y, n_cells_x, bins))

        for i in range(n_cells_y):
            for j in range(n_cells_x):
                cell_mag = magnitude[i*cell_size:(i+1)*cell_size,
                                     j*cell_size:(j+1)*cell_size]
                cell_ang = orientation[i*cell_size:(i+1)*cell_size,
                                       j*cell_size:(j+1)*cell_size]

                hist    = np.zeros(bins)
                bin_idx = np.clip((cell_ang // (180 / bins)).astype(int), 0, bins - 1)
                np.add.at(hist, bin_idx.flatten(), cell_mag.flatten())
                hist_tensor[i, j] = hist

        hog_vector = []
        for i in range(n_cells_y - block_size + 1):
            for j in range(n_cells_x - block_size + 1):
                block = hist_tensor[i:i+block_size, j:j+block_size].flatten()
                block = block / (np.sqrt(np.sum(block**2)) + eps)
                hog_vector.extend(block)

        hog_all.append(hog_vector)

    return np.array(hog_all)


# ── K-Fold ────────────────────────────────────────────────────────────────────

def k_fold_split(X, y, k=5, shuffle=True, random_state=42):
    rng       = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    indices   = np.arange(n_samples)

    if shuffle:
        rng.shuffle(indices)

    fold_sizes = np.full(k, n_samples // k)
    fold_sizes[:n_samples % k] += 1

    current, folds = 0, []
    for fold_size in fold_sizes:
        start, end = current, current + fold_size
        val_idx    = indices[start:end]
        train_idx  = np.concatenate((indices[:start], indices[end:]))
        folds.append((train_idx, val_idx))
        current = end

    return folds


def k_fold_evaluate(X, y, k=5, knn_k=3, feature_fn=None,
                    balance_fn=None, balance_label="none"):
    folds      = k_fold_split(X, y, k)
    accuracies = []

    for i, (train_idx, val_idx) in enumerate(folds):
        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        X_val_fold,   y_val_fold   = X[val_idx],   y[val_idx]

        # Balance training fold only — never touch validation
        if balance_fn is not None:
            X_train_fold, y_train_fold = balance_fn(X_train_fold, y_train_fold)

        # Feature extraction inside fold — prevents leakage
        if feature_fn is not None:
            X_train_fold, X_val_fold = feature_fn(X_train_fold, X_val_fold)

        model = KNN(k=knn_k)
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)

        acc = np.mean(y_pred == y_val_fold)
        accuracies.append(acc)
        print(f"  Fold {i+1} | balance={balance_label} | acc={acc*100:.2f}%")

    return accuracies


# ── Feature function factories for k_fold_evaluate ───────────────────────────

def pca_fn(n_components=30):
    def fn(X_train, X_val):
        X_train_pca, X_val_pca, _ = pca(X_train, X_val, n_components=n_components)
        return X_train_pca, X_val_pca
    return fn


def hog_fn(cell_size=4, block_size=2, bins=9):
    def fn(X_train, X_val):
        return (hog_features(X_train, cell_size, block_size, bins),
                hog_features(X_val,   cell_size, block_size, bins))
    return fn

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