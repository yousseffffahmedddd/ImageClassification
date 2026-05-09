"""
preprocess_ksvm.py  —  Preprocessing for Kernel SVM (OvR RBF)
==============================================================
Pipeline (from kernelsvm_phase2.py  extract_features()):
    1. HOG features           (fit-free pixel descriptor)
       pixels_per_cell=(7,7), cells_per_block=(2,2)
    2. PCA(50)                (fit on X_train only)
    3. hstack(raw_pixels, PCA_features, HOG_features)
    4. StandardScaler         (fit on hstacked X_train only)

Why this combination for a kernel SVM
───────────────────────────────────────
• Raw pixels give the RBF kernel access to global intensity layout.
• PCA(50) captures the principal axes of pixel variance with low
  dimension, helping the kernel focus on dominant structure.
• HOG captures local shape/edge information that raw pixels miss.
• StandardScaler is mandatory: the RBF kernel exp(-γ||x-z||²)
  is sensitive to the absolute scale of every feature dimension.
  Without it, high-variance pixel/HOG dimensions dominate the
  distance computation and make γ tuning unreliable.

Fits nothing on val/test — all transformers are fitted on X_train
then applied identically to val and test.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog as skimage_hog


# ── HOG (same settings as kernelsvm_phase2.py) ───────────────────────
def _extract_hog(X):
    """
    X : (n, 784) float array — raw normalised pixels
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
def fit_transform(X_train, X_val, X_test, n_pca=50):
    """
    Apply full feature pipeline from kernelsvm_phase2.py.

    Parameters
    ----------
    X_train, X_val, X_test : (n, 784) float arrays, pixels in [0, 1]
    n_pca                  : PCA components (default 50)

    Returns
    -------
    X_train_out, X_val_out, X_test_out : (n, 784 + n_pca + hog_dim) arrays,
                                          z-scored
    """
    # Step 1 — HOG (parameter-free, applied to raw pixels)
    print(f"  [KSVM preprocess]  Extracting HOG …")
    hog_tr  = _extract_hog(X_train)
    hog_val = _extract_hog(X_val)
    hog_te  = _extract_hog(X_test)

    # Step 2 — PCA fitted on X_train only
    print(f"  [KSVM preprocess]  PCA({n_pca}) …")
    pca = PCA(n_components=n_pca, random_state=42)
    pca_tr  = pca.fit_transform(X_train)
    pca_val = pca.transform(X_val)
    pca_te  = pca.transform(X_test)
    print(f"  [KSVM preprocess]  Explained variance = "
          f"{pca.explained_variance_ratio_.sum():.3f}")

    # Step 3 — hstack(raw, PCA, HOG)
    X_tr_cat  = np.hstack([X_train, pca_tr,  hog_tr])
    X_val_cat = np.hstack([X_val,   pca_val, hog_val])
    X_te_cat  = np.hstack([X_test,  pca_te,  hog_te])

    # Step 4 — StandardScaler fitted on hstacked X_train only
    scaler = StandardScaler()
    X_train_out = scaler.fit_transform(X_tr_cat)
    X_val_out   = scaler.transform(X_val_cat)
    X_test_out  = scaler.transform(X_te_cat)

    print(f"  [KSVM preprocess]  raw(784) + PCA({n_pca}) + "
          f"HOG({hog_tr.shape[1]}) → scaled dim = {X_train_out.shape[1]}")

    return X_train_out, X_val_out, X_test_out
