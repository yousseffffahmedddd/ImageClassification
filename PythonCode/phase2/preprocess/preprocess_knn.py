"""
preprocess_knn.py  —  Preprocessing for KNN
=============================================
Pipeline (from knn_phase2.py):
    1. PCA(50)  using the manual pca() defined in knn_phase2.py
       (mean-centred, eigen-decomposition of covariance matrix)

Note
----
knn_phase2.py's pca() fits on X_train and projects both splits,
returning (X_train_pca, X_other_pca).  We call it twice — once
for val, once for test — both times fitting on the same X_train_hog
so the projection axes are identical.
"""

import numpy as np


# ── Inline HOG (matches knn_phase2.py's import from knn module) ──────
def _hog_features(images, cell_size=7, block_size=2, bins=9):
    """
    Vectorised HOG matching the version used in the KNN module.
    images : (n, 28, 28) or (n, 784) float array
    """
    if images.ndim == 2:
        n    = images.shape[0]
        side = int(np.sqrt(images.shape[1]))
        images = images.reshape(n, side, side)

    images = images.astype(np.float32)
    n, h, w = images.shape
    eps = 1e-6

    gx = np.zeros_like(images)
    gy = np.zeros_like(images)
    gx[:, :, 1:-1] = images[:, :, 2:] - images[:, :, :-2]
    gy[:, 1:-1, :] = images[:, 2:, :] - images[:, :-2, :]

    magnitude   = np.sqrt(gx**2 + gy**2)
    orientation = (np.arctan2(gy, gx) * 180 / np.pi) % 180

    n_cells_x = w // cell_size
    n_cells_y = h // cell_size

    bin_idx = np.clip(
        (orientation // (180 / bins)).astype(int), 0, bins - 1
    )

    hist_tensor = np.zeros(
        (n, n_cells_y, n_cells_x, bins), dtype=np.float32
    )

    for b in range(bins):
        mask     = (bin_idx == b).astype(np.float32)
        weighted = magnitude * mask
        weighted_cropped = weighted[
            :, :n_cells_y * cell_size, :n_cells_x * cell_size
        ]
        hist_tensor[:, :, :, b] = weighted_cropped.reshape(
            n, n_cells_y, cell_size, n_cells_x, cell_size
        ).sum(axis=(2, 4))

    n_blocks_y    = n_cells_y - block_size + 1
    n_blocks_x    = n_cells_x - block_size + 1
    n_block_feats = block_size * block_size * bins

    hog_matrix = np.zeros(
        (n, n_blocks_y * n_blocks_x * n_block_feats), dtype=np.float32
    )

    block_idx = 0
    for i in range(n_blocks_y):
        for j in range(n_blocks_x):
            block      = hist_tensor[:, i:i+block_size, j:j+block_size, :]
            block_flat = block.reshape(n, -1)
            norms      = np.sqrt((block_flat**2).sum(axis=1, keepdims=True)) + eps
            block_flat = block_flat / norms
            start = block_idx * n_block_feats
            end   = start + n_block_feats
            hog_matrix[:, start:end] = block_flat
            block_idx += 1

    return hog_matrix


# ── Manual PCA (matches knn_phase2.py's pca() exactly) ───────────────
def _pca(X_train, X_other, n_components=50):
    mean_vector      = np.mean(X_train, axis=0)
    X_train_centered = X_train - mean_vector
    X_other_centered = X_other - mean_vector

    cov_matrix                = np.cov(X_train_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_idx  = np.argsort(eigenvalues)[::-1]
    components  = eigenvectors[:, sorted_idx][:, :n_components]

    return (np.dot(X_train_centered, components),
            np.dot(X_other_centered, components))


# ── Public API ────────────────────────────────────────────────────────
def fit_transform(X_train, X_val, X_test, n_pca=50):
    """
    Apply HOG then PCA(50) as in knn_phase2.py.

    Parameters
    ----------
    X_train, X_val, X_test : (n, 784) float arrays, pixels in [0, 1]
    n_pca                  : number of PCA components (default 50)

    Returns
    -------
    X_train_out, X_val_out, X_test_out : (n, n_pca) arrays
    """
    # HOG — parameter-free
    print(f"  [KNN preprocess]  Computing HOG on "
          f"{len(X_train)} / {len(X_val)} / {len(X_test)} samples …")
    X_train_hog = _hog_features(X_train)
    X_val_hog   = _hog_features(X_val)
    X_test_hog  = _hog_features(X_test)

    # PCA — fit on train HOG, project val and test separately
    X_train_out, X_val_out  = _pca(X_train_hog, X_val_hog,  n_pca)
    _,           X_test_out = _pca(X_train_hog, X_test_hog, n_pca)

    print(f"  [KNN preprocess]  HOG({X_train_hog.shape[1]}) "
          f"→ PCA({n_pca})  final dim = {X_train_out.shape[1]}")

    return X_train_out, X_val_out, X_test_out
