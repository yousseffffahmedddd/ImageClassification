import numpy as np
import struct
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from skimage.feature import hog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ══════════════════════════════════════════════════════════════════
#  1. DATA LOADING
#     Reads MNIST binary format (IDX) for images and labels.
# ══════════════════════════════════════════════════════════════════

def load_images(path):
    """Parse IDX3-ubyte image file → (N, 784) uint8 array."""
    with open(path, 'rb') as f:
        _, num, rows, cols = struct.unpack('>IIII', f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)


def load_labels(path):
    """Parse IDX1-ubyte label file → (N,) uint8 array."""
    with open(path, 'rb') as f:
        struct.unpack('>II', f.read(8))          # magic + count (discard)
        return np.frombuffer(f.read(), dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════
#  2. DATA LOADING + SPLITTING
#     Binary task: class 1 = digit "0", class 0 = any other digit.
#     Split: 85 % train / 15 % validation  +  held-out test set.
# ══════════════════════════════════════════════════════════════════

def load_data(dataset_dir):
    """
    Load MNIST, binarise labels (0 vs rest), normalise pixels to [0,1],
    and create train / validation / test splits.
    """
    X_train_raw = load_images(os.path.join(dataset_dir,
                    'train-images-idx3-ubyte', 'train-images-idx3-ubyte'))
    y_train_raw = load_labels(os.path.join(dataset_dir,
                    'train-labels-idx1-ubyte', 'train-labels-idx1-ubyte'))
    X_test_raw  = load_images(os.path.join(dataset_dir,
                    't10k-images-idx3-ubyte',  't10k-images-idx3-ubyte'))
    y_test_raw  = load_labels(os.path.join(dataset_dir,
                    't10k-labels-idx1-ubyte',  't10k-labels-idx1-ubyte'))

    # Binarise: 1 = digit zero, 0 = not zero
    y_train = (y_train_raw == 0).astype(int)
    y_test  = (y_test_raw  == 0).astype(int)

    # Pixel normalisation to [0, 1]
    X_train = X_train_raw / 255.0
    X_test  = X_test_raw  / 255.0

    # Stratified 85/15 train-validation split
    rng       = np.random.default_rng(42)
    idx       = rng.permutation(len(X_train))
    split     = int(0.85 * len(X_train))
    X_tr,  X_val  = X_train[idx[:split]],  X_train[idx[split:]]
    y_tr,  y_val  = y_train[idx[:split]],  y_train[idx[split:]]

    print(f"[Data] Train={len(y_tr)}  Val={len(y_val)}  Test={len(y_test)}")
    print(f"[Data] Train class distribution — 0: {(y_tr==0).sum()}  1: {(y_tr==1).sum()}")
    return X_tr, y_tr, X_val, y_val, X_test, y_test


# ══════════════════════════════════════════════════════════════════
#  3. FEATURE EXTRACTION
#     Three complementary representations are concatenated:
#       a) Flattened pixels  (784-dim) — raw intensity baseline
#       b) PCA components    (50-dim)  — global low-rank structure
#       c) HOG descriptors   (36-dim)  — edge / gradient features
#     All features are then z-score normalised (StandardScaler).
# ══════════════════════════════════════════════════════════════════

def extract_hog_features(X):
    """
    Compute HOG descriptor for each image in X (shape N×784).
    Settings: 7×7 pixels_per_cell, 2×2 cells_per_block → 36-dim vector.
    """
    features = []
    for img in X:
        h = hog(img.reshape(28, 28),
                pixels_per_cell=(7, 7),
                cells_per_block=(2, 2),
                feature_vector=True)
        features.append(h)
    return np.array(features)


def extract_features(X_train, X_val, X_test, n_pca=50):
    """
    Build combined feature matrix [flatten | PCA | HOG] and z-score normalise.
    PCA and StandardScaler are fit ONLY on the training set to prevent leakage.

    Parameters
    ----------
    n_pca : int
        Number of PCA components (default 50 — retains ~85 % variance).
    """
    print(f"[Features] Extracting HOG descriptors …")
    hog_train = extract_hog_features(X_train)
    hog_val   = extract_hog_features(X_val)
    hog_test  = extract_hog_features(X_test)

    print(f"[Features] Fitting PCA (n={n_pca}) …")
    pca = PCA(n_components=n_pca, random_state=42)
    pca_train = pca.fit_transform(X_train)
    pca_val   = pca.transform(X_val)
    pca_test  = pca.transform(X_test)
    print(f"[Features] PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Concatenate: flatten (784) + PCA (50) + HOG (36) = 870 dims
    X_tr_feat  = np.hstack([X_train, pca_train, hog_train])
    X_val_feat = np.hstack([X_val,   pca_val,   hog_val])
    X_te_feat  = np.hstack([X_test,  pca_test,  hog_test])

    # Z-score normalisation — fit ONLY on training data
    scaler = StandardScaler()
    X_tr_feat  = scaler.fit_transform(X_tr_feat)
    X_val_feat = scaler.transform(X_val_feat)
    X_te_feat  = scaler.transform(X_te_feat)

    print(f"[Features] Final feature-vector dimension: {X_tr_feat.shape[1]}")
    return X_tr_feat, X_val_feat, X_te_feat


# ══════════════════════════════════════════════════════════════════
#  4. CLASS-IMBALANCE HANDLING
#     MNIST contains ~10 % zeros (≈5 923 / 60 000).
#     Strategy: random oversampling of the minority class until
#     both classes are equally represented in training data.
# ══════════════════════════════════════════════════════════════════

def handle_imbalance(X, y, seed=42):
    """
    Oversample the minority class to balance the training set.
    Only ever applied to the TRAINING split (never val/test).
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    print(f"[Imbalance] Before — class 0: {counts[0]}  class 1: {counts[1]}")

    max_count = counts.max()
    X_parts, y_parts = [X], [y]
    for cls, cnt in zip(classes, counts):
        if cnt < max_count:
            idx  = np.where(y == cls)[0]
            over = rng.choice(idx, max_count - cnt, replace=True)
            X_parts.append(X[over])
            y_parts.append(y[over])

    X_bal = np.vstack(X_parts)
    y_bal = np.hstack(y_parts)
    shuffle = rng.permutation(len(y_bal))
    print(f"[Imbalance] After  — total: {len(y_bal)} (balanced)")
    return X_bal[shuffle], y_bal[shuffle]


# ══════════════════════════════════════════════════════════════════
#  5. LINEAR SVM  (from scratch — gradient descent on hinge loss)
#
#  Mathematical formulation
#  ─────────────────────────
#  Labels re-coded: ŷ ∈ {−1, +1}
#
#  Hinge (primal) loss:
#    L(w, b) = λ‖w‖² + (1/n) Σᵢ max(0, 1 − yᵢ(w·xᵢ + b))
#
#  Sub-gradient w.r.t. w (S = {i : yᵢ(w·xᵢ+b) < 1}):
#    ∂L/∂w = 2λw − (1/|S|) Σᵢ∈S yᵢxᵢ
#    ∂L/∂b =      − (1/|S|) Σᵢ∈S yᵢ
#
#  Learning-rate schedule (inverse-scaling):
#    ηₜ = η₀ / (1 + η₀ · λ · t)
#
#  Prediction:
#    ŷ = sign(w·x + b),  mapped back to {0, 1}
# ══════════════════════════════════════════════════════════════════

class LinearSVM:
    """
    Binary Linear Support Vector Machine trained with sub-gradient descent
    and an inverse-scaling learning rate schedule.

    Parameters
    ----------
    lr            : float  — initial learning rate η₀
    lambda_param  : float  — L2 regularisation strength λ
    epochs        : int    — number of full passes over training data
    """

    def __init__(self, lr=0.01, lambda_param=0.001, epochs=100):
        self.lr           = lr
        self.lambda_param = lambda_param
        self.epochs       = epochs
        self.loss_history = []
        self.w            = None
        self.b            = 0.0

    # ── training ──────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Fit the SVM to training data.

        Parameters
        ----------
        X : ndarray (n, d)  — feature matrix
        y : ndarray (n,)    — binary labels in {0, 1}
        """
        y_ = np.where(y == 0, -1.0, 1.0)   # recode to {-1, +1}
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for epoch in range(self.epochs):
            # Inverse-scaling schedule: η_t = η₀ / (1 + η₀·λ·t)
            eta = self.lr / (1.0 + self.lr * self.lambda_param * (epoch + 1))

            margins  = y_ * (X @ self.w + self.b)
            violated = margins < 1                        # support-vector set S

            # Sub-gradients
            dw = 2.0 * self.lambda_param * self.w
            db = 0.0
            if violated.any():
                dw -= np.mean(y_[violated, None] * X[violated], axis=0)
                db -= np.mean(y_[violated])

            # Parameter update
            self.w -= eta * dw
            self.b -= eta * db

            # Hinge loss (for monitoring)
            loss = (self.lambda_param * np.dot(self.w, self.w)
                    + np.mean(np.maximum(0.0, 1.0 - margins)))
            self.loss_history.append(float(loss))

    # ── inference ─────────────────────────────────────────────────

    def predict(self, X):
        """Return class labels in {0, 1}."""
        raw = X @ self.w + self.b
        return np.where(np.sign(raw) == -1, 0, 1)

    def decision_function(self, X):
        """Return raw signed distance from the hyperplane."""
        return X @ self.w + self.b


# ══════════════════════════════════════════════════════════════════
#  6. K-FOLD CROSS-VALIDATION
#     Performed on the (balanced) training set only.
#     Validation and test sets are NEVER seen during CV.
# ══════════════════════════════════════════════════════════════════

# def k_fold_cv(X, y, k=5, lr=0.01, lambda_param=0.001, epochs=100):
#     """
#     k-fold cross-validation on training data.

#     Returns
#     -------
#     metrics : list of dicts  — per-fold {acc, prec, rec, f1}
#     """
#     fold_size = len(X) // k
#     metrics   = []

#     print(f"\n[CV] Running {k}-Fold Cross-Validation …")
#     for fold in range(k):
#         val_start = fold * fold_size
#         val_end   = (fold + 1) * fold_size

#         val_idx   = np.arange(val_start, val_end)
#         train_idx = np.concatenate([np.arange(0, val_start),
#                                     np.arange(val_end, len(X))])

#         X_cv_tr, y_cv_tr = X[train_idx], y[train_idx]
#         X_cv_vl, y_cv_vl = X[val_idx],   y[val_idx]

#         model = LinearSVM(lr=lr, lambda_param=lambda_param, epochs=epochs)
#         model.fit(X_cv_tr, y_cv_tr)
#         y_pred = model.predict(X_cv_vl)

#         fold_metrics = {
#             'acc' : accuracy_score (y_cv_vl, y_pred),
#             'prec': precision_score(y_cv_vl, y_pred, zero_division=0),
#             'rec' : recall_score   (y_cv_vl, y_pred, zero_division=0),
#             'f1'  : f1_score       (y_cv_vl, y_pred, zero_division=0),
#         }
#         metrics.append(fold_metrics)
#         print(f"  Fold {fold+1}/{k}  acc={fold_metrics['acc']:.4f}"
#               f"  prec={fold_metrics['prec']:.4f}"
#               f"  rec={fold_metrics['rec']:.4f}"
#               f"  f1={fold_metrics['f1']:.4f}")

#     print("\n[CV] Mean ± Std across folds:")
#     for key in ['acc', 'prec', 'rec', 'f1']:
#         vals = [m[key] for m in metrics]
#         print(f"  {key:4s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

#     return metrics


# # ══════════════════════════════════════════════════════════════════
# #  7. EVALUATION  (on any split)
# # ══════════════════════════════════════════════════════════════════

# def evaluate(model, X, y, name="", save_dir="."):
#     """
#     Compute and print classification metrics; save confusion matrix plot.

#     Returns
#     -------
#     dict with accuracy, precision, recall, f1
#     """
#     y_pred = model.predict(X)
#     acc    = accuracy_score (y, y_pred)
#     prec   = precision_score(y, y_pred, zero_division=0)
#     rec    = recall_score   (y, y_pred, zero_division=0)
#     f1     = f1_score       (y, y_pred, zero_division=0)
#     cm     = confusion_matrix(y, y_pred)

#     print(f"\n── {name} ──────────────────────────────────────")
#     print(f"  Accuracy  : {acc :.4f}")
#     print(f"  Precision : {prec:.4f}")
#     print(f"  Recall    : {rec :.4f}")
#     print(f"  F1-score  : {f1  :.4f}")
#     print(f"  Confusion Matrix:\n{cm}")

#     # ── confusion matrix plot ──────────────────────────────────────
#     fig, ax = plt.subplots(figsize=(4, 3.5))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
#                 xticklabels=['Pred: Not-0', 'Pred: Zero'],
#                 yticklabels=['True: Not-0', 'True: Zero'])
#     ax.set_title(f"Confusion Matrix — {name}", fontsize=11, fontweight='bold')
#     plt.tight_layout()
#     out_path = os.path.join(save_dir, f"cm_{name.lower().replace(' ', '_')}.png")
#     plt.savefig(out_path, dpi=150)
#     plt.close(fig)
#     print(f"  [Plot] Saved → {out_path}")

#     return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


# # ══════════════════════════════════════════════════════════════════
# #  8. PLOTTING UTILITIES
# # ══════════════════════════════════════════════════════════════════

# def plot_loss_curve(loss_history, save_dir="."):
#     """Plot training loss (regularised hinge) vs. epoch."""
#     fig, ax = plt.subplots(figsize=(7, 4))
#     ax.plot(loss_history, color='steelblue', linewidth=1.5)
#     ax.set_xlabel("Epoch", fontsize=12)
#     ax.set_ylabel("Loss  (λ‖w‖² + hinge)", fontsize=12)
#     ax.set_title("Training Loss Curve — Linear SVM", fontsize=13, fontweight='bold')
#     ax.grid(alpha=0.35)
#     plt.tight_layout()
#     path = os.path.join(save_dir, "loss_curve.png")
#     plt.savefig(path, dpi=150)
#     plt.close(fig)
#     print(f"[Plot] Loss curve saved → {path}")


# def plot_cv_metrics(cv_metrics, save_dir="."):
#     """Bar chart of per-fold accuracy and F1 from k-fold CV."""
#     k      = len(cv_metrics)
#     folds  = [f"Fold {i+1}" for i in range(k)]
#     accs   = [m['acc'] for m in cv_metrics]
#     f1s    = [m['f1']  for m in cv_metrics]

#     x   = np.arange(k)
#     w   = 0.35
#     fig, ax = plt.subplots(figsize=(8, 4))
#     ax.bar(x - w/2, accs, w, label='Accuracy', color='steelblue', alpha=0.85)
#     ax.bar(x + w/2, f1s,  w, label='F1-score',  color='darkorange', alpha=0.85)
#     ax.set_xticks(x);  ax.set_xticklabels(folds)
#     ax.set_ylim(0.85, 1.01)
#     ax.set_ylabel("Score", fontsize=12)
#     ax.set_title(f"{k}-Fold CV — Accuracy & F1 per Fold", fontsize=13, fontweight='bold')
#     ax.legend()
#     ax.grid(axis='y', alpha=0.35)
#     plt.tight_layout()
#     path = os.path.join(save_dir, "cv_metrics.png")
#     plt.savefig(path, dpi=150)
#     plt.close(fig)
#     print(f"[Plot] CV metrics saved → {path}")


# def plot_results_comparison(results, save_dir="."):
#     """Grouped bar chart comparing Train / Val / Test metrics."""
#     splits  = list(results.keys())
#     metrics = ['accuracy', 'precision', 'recall', 'f1']
#     colors  = ['steelblue', 'darkorange', 'seagreen', 'crimson']

#     x = np.arange(len(splits))
#     w = 0.18
#     fig, ax = plt.subplots(figsize=(9, 5))
#     for i, (metric, color) in enumerate(zip(metrics, colors)):
#         vals = [results[s][metric] for s in splits]
#         ax.bar(x + (i - 1.5) * w, vals, w, label=metric.capitalize(), color=color, alpha=0.85)

#     ax.set_xticks(x);  ax.set_xticklabels(splits, fontsize=12)
#     ax.set_ylim(0.85, 1.02)
#     ax.set_ylabel("Score", fontsize=12)
#     ax.set_title("Linear SVM — Metric Comparison Across Splits", fontsize=13, fontweight='bold')
#     ax.legend(fontsize=10)
#     ax.grid(axis='y', alpha=0.35)
#     plt.tight_layout()
#     path = os.path.join(save_dir, "results_comparison.png")
#     plt.savefig(path, dpi=150)
#     plt.close(fig)
#     print(f"[Plot] Results comparison saved → {path}")


# def print_results_table(results):
#     """Print a formatted results table to stdout."""
#     header = f"\n{'='*58}"
#     print(header)
#     print(f"{'Split':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
#     print(f"{'-'*58}")
#     for name, m in results.items():
#         print(f"{name:<10} {m['accuracy']:>10.4f} {m['precision']:>10.4f}"
#               f" {m['recall']:>10.4f} {m['f1']:>10.4f}")
#     print(f"{'='*58}")


# # ══════════════════════════════════════════════════════════════════
# #  9. HYPERPARAMETER SEARCH  (grid search on val set)
# #     Evaluates combinations of lr × lambda × epochs and returns
# #     the configuration with the highest validation F1.
# # ══════════════════════════════════════════════════════════════════

def hyperparameter_search(X_tr, y_tr, X_val, y_val):
    """
    Grid search over learning rate, lambda, and epochs.
    Selects best configuration by validation F1-score.
    """
    param_grid = {
        'lr'    : [0.001, 0.005, 0.01],
        'lambda': [0.0001, 0.001, 0.01],
        'epochs': [100, 150],
    }

    best_f1, best_params, best_model = 0.0, {}, None
    print("\n[HyperSearch] Grid search …")

    for lr in param_grid['lr']:
        for lam in param_grid['lambda']:
            for ep in param_grid['epochs']:
                m = LinearSVM(lr=lr, lambda_param=lam, epochs=ep)
                m.fit(X_tr, y_tr)
                f1 = f1_score(y_val, m.predict(X_val), zero_division=0)
                print(f"  lr={lr}  λ={lam}  ep={ep}  val_f1={f1:.4f}")
                if f1 > best_f1:
                    best_f1, best_params, best_model = f1, dict(lr=lr, lam=lam, ep=ep), m

    print(f"\n[HyperSearch] Best → {best_params}  val_f1={best_f1:.4f}")
    return best_params, best_model


# # ══════════════════════════════════════════════════════════════════
# #  MAIN
# # ══════════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     np.random.seed(42)
#     SAVE_DIR    = "."
#     DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                                '..', 'dataset')

#     # ── 1. Load data ────────────────────────────────────────────
#     X_train, y_train, X_val, y_val, X_test, y_test = load_data(DATASET_DIR)

#     # ── 2. Feature extraction ───────────────────────────────────
#     X_train, X_val, X_test = extract_features(X_train, X_val, X_test, n_pca=50)

#     # ── 3. Handle class imbalance (training set only) ───────────
#     X_train, y_train = handle_imbalance(X_train, y_train)

#     # ── 4. Hyperparameter search ────────────────────────────────
#     #best_params, _ = hyperparameter_search(X_train, y_train, X_val, y_val)
#     best_params = {'lr': 0.01, 'lam': 0.01, 'ep': 150}  # pre-computed best config

#     # ── 5. K-fold cross-validation ──────────────────────────────
#     cv_metrics = k_fold_cv(
#         X_train, y_train, k=5,
#         lr=best_params['lr'],
#         lambda_param=best_params['lam'],
#         epochs=best_params['ep'],
#     )
#     plot_cv_metrics(cv_metrics, SAVE_DIR)

#     # ── 6. Train final model on full training set ───────────────
#     print("\n[Train] Fitting final model …")
#     model = LinearSVM(
#         lr=best_params['lr'],
#         lambda_param=best_params['lam'],
#         epochs=best_params['ep'],
#     )
#     model.fit(X_train, y_train)
#     plot_loss_curve(model.loss_history, SAVE_DIR)

#     # ── 7. Evaluate on all splits ───────────────────────────────
#     results = {}
#     results['Train'] = evaluate(model, X_train, y_train, "Train", SAVE_DIR)
#     results['Val']   = evaluate(model, X_val,   y_val,   "Val",   SAVE_DIR)
#     results['Test']  = evaluate(model, X_test,  y_test,  "Test",  SAVE_DIR)

#     # ── 8. Summary ──────────────────────────────────────────────
#     print_results_table(results)
#     plot_results_comparison(results, SAVE_DIR)

#     print("\n[Done] All outputs saved.")
