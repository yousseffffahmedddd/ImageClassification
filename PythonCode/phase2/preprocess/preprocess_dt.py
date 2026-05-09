"""
preprocess_dt.py  —  Preprocessing for Decision Tree
======================================================
Pipeline (from DT_phase2.py):
    1. PCA(150)  (fit on X_train only, applied to val + test)

Why NO StandardScaler
──────────────────────
Decision Trees split on thresholds using Gini impurity.
The split criterion compares feature values to a threshold —
it is purely ordinal.  Scaling (z-score or min-max) is a
monotonic transformation that preserves the ordering of every
feature column, so it cannot change which threshold wins any
Gini split.  StandardScaler is genuinely useless for trees
and is NOT applied.

Why PCA(150)
────────────
Raw MNIST pixels are 784-dimensional and highly sparse.
PCA(150) compresses them into 150 dense orthogonal components
that capture the dominant pixel variance, giving the tree
faster and more informative splits without the noise of
redundant near-zero pixel columns.

DT_phase2.py applies no internal scaling — it expects the
PCA-reduced array directly.
"""

import numpy as np
from sklearn.decomposition import PCA


# ── Public API ────────────────────────────────────────────────────────
def fit_transform(X_train, X_val, X_test, n_pca=150):
    """
    Fit PCA on X_train only; project all three splits.

    Parameters
    ----------
    X_train, X_val, X_test : (n, 784) float arrays, pixels in [0, 1]
    n_pca                  : number of PCA components (default 150)

    Returns
    -------
    X_train_out, X_val_out, X_test_out : (n, n_pca) arrays
    """
    pca = PCA(n_components=n_pca, random_state=42)
    X_train_out = pca.fit_transform(X_train)
    X_val_out   = pca.transform(X_val)
    X_test_out  = pca.transform(X_test)

    print(f"  [DT preprocess]  PCA({n_pca})  "
          f"final dim = {X_train_out.shape[1]}")

    return X_train_out, X_val_out, X_test_out
