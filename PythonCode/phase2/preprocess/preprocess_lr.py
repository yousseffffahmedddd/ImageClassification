"""
preprocess_lr.py  —  Preprocessing for Logistic Regression (Softmax)
======================================================================
Pipeline (from logisticregression_phase2.py):
    1. StandardScaler  (fit on train only)
    2. PCA(90)         (fit on scaled train only)
    3. HOG features    (no fitting — pixel descriptor)
    4. hstack(PCA features, HOG features)

Both train and test go through the same fitted scaler + PCA.
HOG is parameter-free so it is applied independently.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog as skimage_hog


# ── HOG (same settings as logisticregression_phase2.py) ──────────────
def _extract_hog(X):
    """
    X : (n, 784) float array  — raw normalised pixels
    Returns (n, hog_dim) float array
    """
    features = []
    for img in X:
        h = skimage_hog(
            img.reshape(28, 28),
            pixels_per_cell=(7, 7),
            cells_per_block=(2, 2),
            feature_vector=True
        )
        features.append(h)
    return np.array(features)


# ── Public API ────────────────────────────────────────────────────────
def fit_transform(X_train, X_val, X_test, n_pca=90):
    """
    Fit scaler + PCA on X_train; apply to all three splits.
    HOG is applied independently (no fitting needed).

    Parameters
    ----------
    X_train, X_val, X_test : (n, 784) float arrays, pixels in [0, 1]
    n_pca                  : number of PCA components (default 90)

    Returns
    -------
    X_train_out, X_val_out, X_test_out : (n, n_pca + hog_dim) arrays
    """
    # Step 1 — StandardScaler
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    # Step 2 — PCA
    pca = PCA(n_components=n_pca, random_state=42)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_val_pca   = pca.transform(X_val_sc)
    X_test_pca  = pca.transform(X_test_sc)

    # Step 3 — HOG  (applied to raw scaled pixels, not PCA output)
    X_train_hog = _extract_hog(X_train_sc)
    X_val_hog   = _extract_hog(X_val_sc)
    X_test_hog  = _extract_hog(X_test_sc)

    # Step 4 — Combine
    X_train_out = np.hstack([X_train_pca, X_train_hog])
    X_val_out   = np.hstack([X_val_pca,   X_val_hog])
    X_test_out  = np.hstack([X_test_pca,  X_test_hog])

    print(f"  [LR preprocess]  "
          f"PCA({n_pca}) + HOG({X_train_hog.shape[1]}) "
          f"→ combined dim = {X_train_out.shape[1]}")

    return X_train_out, X_val_out, X_test_out
