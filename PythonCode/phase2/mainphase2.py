"""
mainphase2.py  —  Full 5-Fold CV Pipeline for MNIST Multi-Class
================================================================
Preprocessing is delegated to one module per model:

    preprocess_lr.py   →  StandardScaler → PCA(90) + HOG  → hstack
    preprocess_knn.py  →  HOG → PCA(50)
    preprocess_dt.py   →  PCA(150) only  (no scaler — trees are scale-invariant)
    preprocess_nb.py   →  HOG → PCA(100)
    preprocess_svm.py  →  StandardScaler → raw 784-dim
    preprocess_ksvm.py →  raw + PCA(50) + HOG → hstack → StandardScaler

All metrics (Accuracy, Precision, Recall, F1) are computed from
scratch using NumPy — no sklearn.metrics anywhere.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Model imports ─────────────────────────────────────────────────────
from logisticregression import LogisticRegression
from knn                import KNN
from DT                 import DecisionTree
from naivebayes         import GaussianNB
from linearsvm          import LinearSVM
from kernelsvm          import OvRKernelSVM

# ── Preprocessing modules (one per model) ────────────────────────────
import preprocess_lr
import preprocess_knn
import preprocess_dt
import preprocess_nb
import preprocess_svm
import preprocess_ksvm


# ══════════════════════════════════════════════════════════════════════
#  1. LOAD MNIST
# ══════════════════════════════════════════════════════════════════════
base_path = "C:/Users/Lenovo/Desktop/mnist/project/dataset/"

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)

x_train_raw = load_mnist_images(
    base_path + 'train-images-idx3-ubyte/train-images-idx3-ubyte')
y_train_raw = load_mnist_labels(
    base_path + 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
x_test_raw  = load_mnist_images(
    base_path + 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
y_test_raw  = load_mnist_labels(
    base_path + 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

print("Raw shapes:", x_train_raw.shape, y_train_raw.shape,
                    x_test_raw.shape,  y_test_raw.shape)


# ══════════════════════════════════════════════════════════════════════
#  2. NORMALISE
# ══════════════════════════════════════════════════════════════════════
X_all        = x_train_raw.reshape(-1, 784) / 255.0   # (60000, 784)
y_all        = y_train_raw.astype(int)
X_test_final = x_test_raw.reshape(-1, 784)  / 255.0   # (10000, 784)
y_test_final = y_test_raw.astype(int)


# ══════════════════════════════════════════════════════════════════════
#  3. STRATIFIED 5-FOLD CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════
K_FOLDS = 5

def stratified_kfold(X, y, k=5, seed=42):
    """
    Yields (X_train_fold, y_train_fold, X_val_fold, y_val_fold, fold_num)
    for each of the k folds, ensuring each class is proportionally
    represented in every fold.
    """
    np.random.seed(seed)
    classes      = np.unique(y)
    class_chunks = {}

    for cls in classes:
        idx = np.where(y == cls)[0]
        np.random.shuffle(idx)
        class_chunks[cls] = np.array_split(idx, k)

    for fold in range(k):
        val_idx, train_idx = [], []

        for cls in classes:
            chunks = class_chunks[cls]
            val_idx.extend(chunks[fold])
            for j, chunk in enumerate(chunks):
                if j != fold:
                    train_idx.extend(chunk)

        val_idx   = np.array(val_idx)
        train_idx = np.array(train_idx)
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)

        yield (X[train_idx], y[train_idx],
               X[val_idx],   y[val_idx],
               fold + 1)


print(f"\n── 5-Fold CV summary ──────────────────────────────────────")
print(f"  Full training pool  : {len(y_all):6d} samples")
print(f"  Per fold approx     :  train ~{int(len(y_all)*0.8):5d}  "
      f"val ~{int(len(y_all)*0.2):5d}  (80 / 20 each fold)")
print(f"  Test set (held-out) : {len(y_test_final):6d} samples")
print(f"──────────────────────────────────────────────────────────")


# ══════════════════════════════════════════════════════════════════════
#  4. PREPROCESSING DISPATCHER
#     Calls the correct preprocess_<model>.py module.
#     Each module's fit_transform(X_train, X_val, X_test) fits ONLY
#     on X_train and applies the fitted transform to X_val / X_test.
# ══════════════════════════════════════════════════════════════════════
def preprocess(model_name, X_tr, X_val, X_te):
    """
    Dispatch to the correct preprocessing module.

    Returns
    -------
    X_tr_p, X_val_p, X_te_p  : transformed arrays ready for the model
    """
    if model_name == 'Logistic Regression':
        return preprocess_lr.fit_transform(X_tr, X_val, X_te)

    elif model_name == 'KNN':
        return preprocess_knn.fit_transform(X_tr, X_val, X_te)

    elif model_name == 'Decision Tree':
        return preprocess_dt.fit_transform(X_tr, X_val, X_te)

    elif model_name == 'Naive Bayes':
        return preprocess_nb.fit_transform(X_tr, X_val, X_te)

    elif model_name == 'SVM':
        return preprocess_svm.fit_transform(X_tr, X_val, X_te)

    elif model_name == 'Kernel SVM':
        return preprocess_ksvm.fit_transform(X_tr, X_val, X_te)

    else:
        raise ValueError(f"Unknown model name: '{model_name}'")


# ══════════════════════════════════════════════════════════════════════
#  5. METRICS  (pure NumPy — no sklearn)
# ══════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred):
    """
    Compute Accuracy, macro Precision, macro Recall, macro F1,
    and the full confusion matrix — all from scratch.

    Returns
    -------
    acc   : float
    prec  : float  (macro average)
    rec   : float  (macro average)
    f1    : float  (macro average)
    cm    : (n_classes, n_classes) int array
    """
    y_true  = np.array(y_true, dtype=int)
    y_pred  = np.array(y_pred, dtype=int)
    classes = np.unique(y_true)

    # Overall accuracy
    acc = float(np.mean(y_true == y_pred))

    # Per-class precision / recall / F1
    precs, recs, f1s = [], [], []
    for cls in classes:
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))

        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = (2 * p * r) / (p + r) if (p + r) else 0.0

        precs.append(p)
        recs.append(r)
        f1s.append(f)

    prec = float(np.mean(precs))
    rec  = float(np.mean(recs))
    f1   = float(np.mean(f1s))

    # Confusion matrix
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            cm[i, j] = int(
                np.sum((y_true == true_cls) & (y_pred == pred_cls))
            )

    return acc, prec, rec, f1, cm


# ══════════════════════════════════════════════════════════════════════
#  6. CONFUSION MATRIX PLOT
# ══════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(cm, model_name, condition, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)

    n      = cm.shape[0]
    labels = [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.4, linecolor='white',
                annot_kws={'size': 8})

    acc = cm.diagonal().sum() / cm.sum()
    ax.set_title(
        f"{model_name}  [{condition}]\nAcc={acc:.4f}",
        fontsize=11, pad=12
    )
    ax.set_xlabel('Predicted Digit', fontsize=10)
    ax.set_ylabel('True Digit',      fontsize=10)

    safe = model_name.replace(' ', '_')
    path = os.path.join(save_dir, f"cm_{safe}_{condition}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"    Saved → {path}")


# ══════════════════════════════════════════════════════════════════════
#  7. MODEL CLONING
# ══════════════════════════════════════════════════════════════════════
def clone_model(model):
    """Return a fresh model instance with the same constructor params."""
    if hasattr(model, 'get_params'):
        return type(model)(**model.get_params())
    return type(model)()


# ══════════════════════════════════════════════════════════════════════
#  8. MAIN PIPELINE  —  5-FOLD CV  +  FINAL TEST EVALUATION
#
#  Step A — CV loop (5 folds)
#    Each fold:
#      • Preprocessing fitted FRESH on that fold's training portion only
#      • Fresh model clone — no weight leakage between folds
#      • Record Acc / Prec / Rec / F1 for this fold
#    After 5 folds: report mean ± std
#
#  Step B — Final model
#    • Retrain on all 60 000 training samples
#    • Preprocessing fitted on all 60 000
#    • Evaluate ONCE on X_test_final (10 000 held-out)
#    • Plot confusion matrix
# ══════════════════════════════════════════════════════════════════════
def run_pipeline(model, model_name, X_all_tr, y_all_tr,
                 X_te, y_te, condition='Imbalanced'):

    print(f"\n  ── {model_name}  [{condition}]")
    print(f"     CV pool : {len(y_all_tr)} samples")

    # ── Step A: 5-Fold CV ─────────────────────────────────────────────
    fold_accs, fold_precs, fold_recs, fold_f1s = [], [], [], []

    for X_tr_f, y_tr_f, X_val_f, y_val_f, fold_num in \
            stratified_kfold(X_all_tr, y_all_tr, k=K_FOLDS):

        fold_model = clone_model(model)

        # Preprocessing fitted ONLY on this fold's train portion
        X_tr_p, X_val_p, _ = preprocess(model_name, X_tr_f, X_val_f, X_te)

        fold_model.fit(X_tr_p, y_tr_f)
        y_val_pred = fold_model.predict(X_val_p)

        acc, prec, rec, f1, _ = compute_metrics(y_val_f, y_val_pred)
        fold_accs.append(acc)
        fold_precs.append(prec)
        fold_recs.append(rec)
        fold_f1s.append(f1)

        print(f"     Fold {fold_num}/5 →  "
              f"Acc={acc:.4f}  Prec={prec:.4f}  "
              f"Rec={rec:.4f}  F1={f1:.4f}")

    # CV summary
    cv_acc   = np.mean(fold_accs);   cv_acc_std  = np.std(fold_accs)
    cv_prec  = np.mean(fold_precs);  cv_prec_std = np.std(fold_precs)
    cv_rec   = np.mean(fold_recs);   cv_rec_std  = np.std(fold_recs)
    cv_f1    = np.mean(fold_f1s);    cv_f1_std   = np.std(fold_f1s)

    print(f"     {'─'*60}")
    print(f"     CV mean →  "
          f"Acc={cv_acc:.4f}±{cv_acc_std:.4f}  "
          f"Prec={cv_prec:.4f}±{cv_prec_std:.4f}  "
          f"Rec={cv_rec:.4f}±{cv_rec_std:.4f}  "
          f"F1={cv_f1:.4f}±{cv_f1_std:.4f}")

    # ── Step B: Final model on full training data ─────────────────────
    final_model = clone_model(model)

    # Dummy single-row val so preprocess signature is satisfied;
    # only the train and test outputs are used here.
    X_all_tr_p, _, X_te_p = preprocess(
        model_name, X_all_tr, X_all_tr[:1], X_te
    )

    final_model.fit(X_all_tr_p, y_all_tr)
    y_te_pred = final_model.predict(X_te_p)

    t_acc, t_prec, t_rec, t_f1, cm = compute_metrics(y_te, y_te_pred)

    print(f"     Test →  "
          f"Acc={t_acc:.4f}  Prec={t_prec:.4f}  "
          f"Rec={t_rec:.4f}  F1={t_f1:.4f}")

    plot_confusion_matrix(cm, model_name, condition)

    return {
        'cv':     (cv_acc,  cv_prec,  cv_rec,  cv_f1),
        'cv_std': (cv_acc_std, cv_prec_std, cv_rec_std, cv_f1_std),
        'test':   (t_acc,  t_prec,  t_rec,  t_f1),
        'cm':     cm
    }


# ══════════════════════════════════════════════════════════════════════
#  9. MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════
def make_models():
    return {
        'Logistic Regression': LogisticRegression(lr=0.1, epochs=300),
        'KNN':                 KNN(),
        'Decision Tree':       DecisionTree(),
        'Naive Bayes':         GaussianNB(),
        'SVM':                 LinearSVM(),
        'Kernel SVM':          OvRKernelSVM(C=1.0, kernel='rbf',
                                             gamma=0.01, max_iter=100,
                                             tol=1e-3),
    }


# ══════════════════════════════════════════════════════════════════════
#  10. RUN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  EXPERIMENT — MULTI-CLASS MNIST (digits 0–9)")
print("  5-fold CV on full 60 000 training pool")
print("═"*65)

all_results = {}

models = make_models()
for name, mdl in models.items():
    r = run_pipeline(
        mdl, name,
        X_all,        y_all,
        X_test_final, y_test_final,
        condition='Imbalanced'
    )
    all_results[name] = {'imbalanced': r}


# ══════════════════════════════════════════════════════════════════════
#  11. SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════
def print_summary(all_results):
    sep = "─" * 95
    hdr = (f"{'Model':<22} "
           f"{'CV Acc':>9} {'CV Prec':>9} {'CV Rec':>9} {'CV F1':>9} "
           f"{'Test Acc':>9} {'Test Prec':>9} {'Test Rec':>9} {'Test F1':>9}")

    print("\n\n" + "═"*95)
    print("  FINAL RESULTS SUMMARY — MULTI-CLASS (digits 0–9)")
    print("  CV metrics = mean across 5 folds  |  Test = held-out 10,000 samples")
    print("═"*95)
    print(hdr)
    print(sep)

    for name, res in all_results.items():
        cv_acc,  cv_prec,  cv_rec,  cv_f1  = res['imbalanced']['cv']
        t_acc,   t_prec,   t_rec,   t_f1   = res['imbalanced']['test']

        print(f"  {name:<22} "
              f"{cv_acc:9.4f} {cv_prec:9.4f} {cv_rec:9.4f} {cv_f1:9.4f} "
              f"{t_acc:9.4f} {t_prec:9.4f} {t_rec:9.4f} {t_f1:9.4f}")

    print(sep)

print_summary(all_results)


# ══════════════════════════════════════════════════════════════════════
#  12. GROUPED BAR CHART
# ══════════════════════════════════════════════════════════════════════
def plot_comparison(all_results, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)

    model_names = list(all_results.keys())
    metrics     = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    color       = '#E74C3C'

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flat

    for ax, metric_name in zip(axes, metrics):
        m_idx = metrics.index(metric_name)
        x     = np.arange(len(model_names))
        w     = 0.5

        vals = [all_results[n]['imbalanced']['test'][m_idx]
                for n in model_names]
        bars = ax.bar(x, vals, w, color=color, alpha=0.85,
                      edgecolor='white', linewidth=0.8)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + 0.005, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=7.5)

        ax.set_title(metric_name, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=18, ha='right', fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel('Score')
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle('Model Comparison — MNIST Multi-Class',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\nComparison chart saved → {path}")

plot_comparison(all_results)


# ══════════════════════════════════════════════════════════════════════
#  13. SIDE-BY-SIDE CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════════
def plot_all_cms(all_results, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    model_names = list(all_results.keys())
    n_models    = len(model_names)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for col, name in enumerate(model_names):
        ax     = axes[col]
        cm     = all_results[name]['imbalanced']['cm']
        labels = [str(i) for i in range(cm.shape[0])]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.4, linecolor='white',
                    annot_kws={'size': 7}, cbar=False)

        f1 = all_results[name]['imbalanced']['test'][3]
        ax.set_title(f"{name}\nF1={f1:.3f}", fontsize=9)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    plt.suptitle('Confusion Matrices — Multi-Class (0–9)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'all_confusion_matrices.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"All-CM figure saved → {path}")

plot_all_cms(all_results)
