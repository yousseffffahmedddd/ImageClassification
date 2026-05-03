import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

from logisticregression import LogisticRegression
from knn               import KNN
from DT                import DecisionTree
from naivebayes        import GaussianNB
from linearsvm         import LinearSVM


# ══════════════════════════════════════════════════════════════
#  1. LOAD MNIST
# ══════════════════════════════════════════════════════════════
base_path = "C:/Users/Lenovo/Desktop/mnist/project/dataset/"

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)

x_train_raw = load_mnist_images(base_path + 'train-images-idx3-ubyte/train-images-idx3-ubyte')
y_train_raw = load_mnist_labels(base_path + 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
x_test_raw  = load_mnist_images(base_path + 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
y_test_raw  = load_mnist_labels(base_path + 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

print("Raw shapes:", x_train_raw.shape, y_train_raw.shape,
                    x_test_raw.shape,  y_test_raw.shape)


# ══════════════════════════════════════════════════════════════
#  2. NORMALISE  (all sets)
# ══════════════════════════════════════════════════════════════
X_all   = x_train_raw.reshape(-1, 784) / 255.0   # (60000, 784)
y_all   = (y_train_raw == 0).astype(int)          # 1 = digit 0, 0 = others

X_test_final = x_test_raw.reshape(-1, 784) / 255.0   # (10000, 784) — NEVER touched
y_test_final = (y_test_raw == 0).astype(int)           # until final evaluation


# ══════════════════════════════════════════════════════════════
#  3. STRATIFIED 5-FOLD CROSS-VALIDATION
#     — the full 60 000 training set is split into 5 equal folds
#     — each fold takes a turn as the validation set (~20%)
#     — the other 4 folds (~80%) are used for training each time
#     — test set (10 000) is NEVER touched until final evaluation
#
#  WHY stratified folds?
#  ─────────────────────
#  With a 1:9 imbalance, a plain shuffle could put almost no
#  digit-0 samples in one fold by chance.  Stratified splitting
#  guarantees the same class ratio inside every fold, making
#  per-fold metrics comparable and reliable.
#
#  WHY 5-fold CV instead of a single 85/15 split?
#  ────────────────────────────────────────────────
#  A single split gives ONE validation score that depends on
#  which samples happened to land in that split.  5-fold CV
#  gives FIVE scores on five different validation sets.
#  The mean ± std across folds is a much more stable and
#  trustworthy estimate of generalisation performance.
# ══════════════════════════════════════════════════════════════
K_FOLDS = 5

def stratified_kfold(X, y, k=5, seed=42):
    """
    Generator that yields (X_train_fold, y_train_fold,
                           X_val_fold,   y_val_fold,
                           fold_number)
    for each of the k folds.

    Algorithm
    ─────────
    1. For each class, shuffle its indices independently.
    2. Split each class's indices into k roughly equal chunks.
    3. Fold i  →  val   = chunk i  from every class (concatenated)
               →  train = all other chunks from every class
    4. Shuffle the combined train and val so the model never
       sees class-ordering artefacts.
    """
    np.random.seed(seed)
    classes      = np.unique(y)
    class_chunks = {}               # {class_label: [chunk_0, ..., chunk_k-1]}

    for cls in classes:
        idx = np.where(y == cls)[0]
        np.random.shuffle(idx)
        class_chunks[cls] = np.array_split(idx, k)

    for fold in range(k):
        val_idx, train_idx = [], []

        for cls in classes:
            chunks = class_chunks[cls]
            val_idx.extend(chunks[fold])                  # this fold → val
            for j, chunk in enumerate(chunks):
                if j != fold:
                    train_idx.extend(chunk)               # rest       → train

        val_idx   = np.array(val_idx)
        train_idx = np.array(train_idx)

        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)

        yield (X[train_idx], y[train_idx],
               X[val_idx],   y[val_idx],
               fold + 1)                                  # 1-based fold number


print(f"\n── 5-Fold CV summary ──────────────────────────────────────")
print(f"  Full training pool  : {len(y_all):6d} samples")
print(f"  Per fold approx     :  train ~{int(len(y_all)*0.8):5d}  "
      f"val ~{int(len(y_all)*0.2):5d}  (80 / 20 each fold)")
print(f"  Test set (held-out) : {len(y_test_final):6d} samples  "
      f"pos={y_test_final.sum()}  neg={(y_test_final==0).sum()}")
print(f"──────────────────────────────────────────────────────────")


# ══════════════════════════════════════════════════════════════
#  4. STANDARD SCALER  (manual — no sklearn dependency)
# ══════════════════════════════════════════════════════════════
class StandardScaler:
    """
    z = (x − μ) / σ
    Fitted ONLY on training data. Same μ and σ applied to val/test.
    """
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_  = np.std(X,  axis=0) + 1e-8   # +1e-8 avoids ÷0 on blank pixels

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ══════════════════════════════════════════════════════════════
#  5. BALANCE  (downsample majority class)
# ══════════════════════════════════════════════════════════════
def downsample_to_balance(X, y, seed=42):
    """
    Reduces the majority class (others) to match minority class (digit 0).
    Labels are carried from the real y array — never manufactured.
    """
    np.random.seed(seed)
    X_pos = X[y == 1];  y_pos = y[y == 1]   # digit 0
    X_neg = X[y == 0];  y_neg = y[y == 0]   # all others

    n = len(y_pos)
    idx = np.random.choice(len(X_neg), size=n, replace=False)

    X_bal = np.vstack([X_pos, X_neg[idx]])
    y_bal = np.hstack([y_pos, y_neg[idx]])   # real labels, not hardcoded

    perm = np.random.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


# ══════════════════════════════════════════════════════════════
#  6. PREPROCESSING  (per model — feature extraction)
# ══════════════════════════════════════════════════════════════
def preprocess(model_name, X_tr, X_val, X_te):
    """
    Each model gets the feature representation best suited to it.

    ┌─────────────────────┬──────────────────────────────────────┐
    │ Model               │ Features & Reason                    │
    ├─────────────────────┼──────────────────────────────────────┤
    │ LogisticRegression  │ StandardScaler → full 784-dim        │
    │                     │ Linear model; z-score stabilises GD  │
    ├─────────────────────┼──────────────────────────────────────┤
    │ KNN                 │ HOG → PCA(50)                        │
    │                     │ Distance-based; HOG distances are    │
    │                     │ shape-aware, not just pixel-wise     │
    ├─────────────────────┼──────────────────────────────────────┤
    │ DecisionTree        │ StandardScaler → PCA(80)             │
    │                     │ Compact dense features give better   │
    │                     │ Gini splits than sparse raw pixels   │
    ├─────────────────────┼──────────────────────────────────────┤
    │ GaussianNB          │ PCA(100) only (no scaler)            │
    │                     │ PCA components are orthogonal —      │
    │                     │ closest to NB independence assumption│
    ├─────────────────────┼──────────────────────────────────────┤
    │ SVM                 │ StandardScaler → full 784-dim        │
    │                     │ SVM objective ||w||² is scale-       │
    │                     │ sensitive; z-score is mandatory      │
    └─────────────────────┴──────────────────────────────────────┘
    """
    from sklearn.decomposition import PCA

    if model_name == 'GaussianNB':
        pca = PCA(n_components=100, random_state=42)
        X_tr_p = pca.fit_transform(X_tr)
        X_val_p = pca.transform(X_val)
        X_te_p = pca.transform(X_te)
        return X_tr_p, X_val_p, X_te_p, pca

    # All other models: StandardScaler first
    sc = StandardScaler()
    X_tr_sc  = sc.fit_transform(X_tr)
    X_val_sc = sc.transform(X_val)
    X_te_sc  = sc.transform(X_te)

    if model_name == 'KNN':
        from knn import hog_features, pca as knn_pca
        X_tr_h  = hog_features(X_tr.reshape(-1, 28, 28))
        X_val_h = hog_features(X_val.reshape(-1, 28, 28))
        X_te_h  = hog_features(X_te.reshape(-1, 28, 28))
        X_tr_sc, X_val_sc, _ = knn_pca(X_tr_h,  X_val_h, 50)
        X_val_sc = X_val_sc
        _, X_te_sc, _ = knn_pca(X_tr_h, X_te_h, 50)

    elif model_name == 'DecisionTree':
        pca = PCA(n_components=80, random_state=42)
        X_tr_sc  = pca.fit_transform(X_tr_sc)
        X_val_sc = pca.transform(X_val_sc)
        X_te_sc  = pca.transform(X_te_sc)

    return X_tr_sc, X_val_sc, X_te_sc


# ══════════════════════════════════════════════════════════════
#  7. METRICS
# ══════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))

    acc  = float(np.mean(y_true == y_pred))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

    return acc, prec, rec, f1, tp, fn, tn, fp


# ══════════════════════════════════════════════════════════════
#  8. CONFUSION MATRIX PLOT
# ══════════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, model_name, condition, save_dir='plots'):
    """
    Plots a 2×2 annotated heatmap confusion matrix and saves to disk.
    condition: 'Imbalanced' or 'Balanced' — used in filename and title.
    """
    os.makedirs(save_dir, exist_ok=True)

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    cm = np.array([[tp, fn],
                   [fp, tn]])

    total = cm.sum()
    annot = np.array([[f"{tp}\n({tp/total*100:.1f}%)", f"{fn}\n({fn/total*100:.1f}%)"],
                      [f"{fp}\n({fp/total*100:.1f}%)", f"{tn}\n({tn/total*100:.1f}%)"]])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax,
                xticklabels=['Pred: Digit 0', 'Pred: Other'],
                yticklabels=['True: Digit 0', 'True: Other'],
                linewidths=1, linecolor='white',
                cbar_kws={'label': 'Count'})

    acc  = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

    ax.set_title(
        f"{model_name}  [{condition}]\n"
        f"Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}",
        fontsize=11, pad=12
    )
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label',      fontsize=10)

    safe = model_name.replace(' ', '_')
    path = os.path.join(save_dir, f"cm_{safe}_{condition}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"    Saved → {path}")


# ══════════════════════════════════════════════════════════════
#  9. CLONE MODEL
# ══════════════════════════════════════════════════════════════
def clone_model(model):
    if hasattr(model, 'get_params'):
        return type(model)(**model.get_params())
    return type(model)()


# ══════════════════════════════════════════════════════════════
#  10. MAIN PIPELINE  —  5-FOLD CV + FINAL TEST EVALUATION
#
#  Flow for each model
#  ───────────────────
#  Step A  (CV loop — 5 iterations)
#    Each iteration:
#      • fold k  →  val  set  (~12 000 samples, ~20%)
#      • other 4 →  train set (~48 000 samples, ~80%)
#      • preprocess is fitted FRESH inside each fold on that
#        fold's training portion only  (no leakage across folds)
#      • record Acc / Prec / Rec / F1 for this fold
#    After 5 iterations: print mean ± std across folds
#
#  Step B  (final model)
#    • Retrain on the ENTIRE X_all (all 60 000 train samples)
#    • Preprocess fitted on all 60 000
#    • Evaluate ONCE on X_test_final (10 000 held-out)
#    • Plot confusion matrix on test set
#    • These test metrics are the official reported results
# ══════════════════════════════════════════════════════════════
def run_pipeline(model, model_name, X_all_tr, y_all_tr,
                 X_te, y_te, condition):

    print(f"\n  ── {model_name}  [{condition}]")
    print(f"     CV pool : {len(y_all_tr)} samples  "
          f"pos={y_all_tr.sum()}  neg={(y_all_tr==0).sum()}")

    # ── Step A: 5-Fold Cross-Validation ──────────────────────
    fold_accs, fold_precs, fold_recs, fold_f1s = [], [], [], []

    for X_tr_f, y_tr_f, X_val_f, y_val_f, fold_num in \
            stratified_kfold(X_all_tr, y_all_tr, k=K_FOLDS):

        # Fresh model clone for each fold — no weight leakage between folds
        fold_model = clone_model(model)

        # Preprocess fitted ONLY on this fold's training portion
        X_tr_p, X_val_p, _ = preprocess(model_name, X_tr_f, X_val_f, X_te)

        fold_model.fit(X_tr_p, y_tr_f)
        y_val_pred = fold_model.predict(X_val_p)

        acc, prec, rec, f1, *_ = compute_metrics(y_val_f, y_val_pred)
        fold_accs.append(acc);   fold_precs.append(prec)
        fold_recs.append(rec);   fold_f1s.append(f1)

        print(f"     Fold {fold_num}/5 →  "
              f"Acc={acc:.4f}  Prec={prec:.4f}  "
              f"Rec={rec:.4f}  F1={f1:.4f}")

    # CV summary
    cv_acc  = np.mean(fold_accs);  cv_acc_std  = np.std(fold_accs)
    cv_prec = np.mean(fold_precs); cv_prec_std = np.std(fold_precs)
    cv_rec  = np.mean(fold_recs);  cv_rec_std  = np.std(fold_recs)
    cv_f1   = np.mean(fold_f1s);   cv_f1_std   = np.std(fold_f1s)

    print(f"     {'─'*56}")
    print(f"     CV mean  →  "
          f"Acc={cv_acc:.4f}±{cv_acc_std:.4f}  "
          f"Prec={cv_prec:.4f}±{cv_prec_std:.4f}  "
          f"Rec={cv_rec:.4f}±{cv_rec_std:.4f}  "
          f"F1={cv_f1:.4f}±{cv_f1_std:.4f}")

    # ── Step B: Final model on all training data → test set ──
    final_model = clone_model(model)
    X_all_tr_p, _, X_te_p = preprocess(model_name, X_all_tr,
                                        X_all_tr[:1], X_te)
    # Note: the middle array (val placeholder) is unused for Step B;
    # we pass X_all_tr[:1] as a dummy so preprocess signature is satisfied.

    final_model.fit(X_all_tr_p, y_all_tr)
    y_te_pred = final_model.predict(X_te_p)

    t_acc, t_prec, t_rec, t_f1, tp, fn, tn, fp = compute_metrics(y_te, y_te_pred)

    print(f"     {'─'*56}")
    print(f"     FINAL TEST  →  "
          f"Acc={t_acc:.4f}  Prec={t_prec:.4f}  "
          f"Rec={t_rec:.4f}  F1={t_f1:.4f}")
    print(f"     CM  →  TP={tp}  FN={fn}  FP={fp}  TN={tn}")

    plot_confusion_matrix(y_te, y_te_pred, model_name, condition)

    return {
        'cv':   (cv_acc, cv_prec, cv_rec, cv_f1),
        'cv_std': (cv_acc_std, cv_prec_std, cv_rec_std, cv_f1_std),
        'test': (t_acc, t_prec, t_rec, t_f1),
        'cm':   (tp, fn, fp, tn)
    }


# ══════════════════════════════════════════════════════════════
#  11. MODEL REGISTRY
# ══════════════════════════════════════════════════════════════
def make_models():
    return {
        # 'Logistic Regression': LogisticRegression(lr=0.1, epochs=300),
        # 'KNN':                 KNN(),
        # 'Decision Tree':       DecisionTree(),
        'Naive Bayes':         GaussianNB(),
        # 'SVM':                 LinearSVM(),
    }


# ══════════════════════════════════════════════════════════════
#  12. RUN BOTH EXPERIMENTS
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  EXPERIMENT 1 — IMBALANCED  (~1:9 ratio)")
print("  5-fold CV on full 60 000 training pool")
print("═"*65)

all_results = {}   # {model_name: {'imbalanced': {...}, 'balanced': {...}}}

imbalanced_models = make_models()
for name, mdl in imbalanced_models.items():
    r = run_pipeline(
        mdl, name,
        X_all,        y_all,          # full 60k — CV splits it internally
        X_test_final, y_test_final,
        condition='Imbalanced'
    )
    all_results.setdefault(name, {})['imbalanced'] = r


# Balance the entire 60k training pool, then run CV on the balanced version
X_all_bal, y_all_bal = downsample_to_balance(X_all, y_all)

print("\n" + "═"*65)
print(f"  EXPERIMENT 2 — BALANCED  (downsampled train)")
print(f"  Balanced pool: {len(y_all_bal)} samples  "
      f"pos={y_all_bal.sum()}  neg={(y_all_bal==0).sum()}")
print("═"*65)

balanced_models = make_models()
for name, mdl in balanced_models.items():
    r = run_pipeline(
        mdl, name,
        X_all_bal,    y_all_bal,      # balanced pool — CV splits it internally
        X_test_final, y_test_final,   # same held-out test set
        condition='Balanced'
    )
    all_results[name]['balanced'] = r


# ══════════════════════════════════════════════════════════════
#  13. SUMMARY TABLE
# ══════════════════════════════════════════════════════════════
def print_summary(all_results):
    sep = "─" * 95
    hdr = (f"{'Model':<22} {'Condition':<12} "
           f"{'CV Acc':>9} {'CV Prec':>9} {'CV Rec':>9} {'CV F1':>9} "
           f"{'Test Acc':>9} {'Test Rec':>9} {'Test Prec':>9} {'Test F1':>9}")

    print("\n\n" + "═"*95)
    print("  FINAL RESULTS SUMMARY")
    print("  CV metrics = mean across 5 folds  |  Test = held-out 10,000 samples")
    print("═"*95)
    print(hdr)
    print(sep)

    for name, res in all_results.items():
        for cond in ['imbalanced', 'balanced']:
            cv_acc, cv_prec, cv_rec, cv_f1 = res[cond]['cv']
            t_acc, t_prec, t_rec, t_f1     = res[cond]['test']

            print(f"  {name:<22} {cond.capitalize():<12} "
                  f"{cv_acc:9.4f} {cv_prec:9.4f} {cv_rec:9.4f} {cv_f1:9.4f} "
                  f"{t_acc:9.4f} {t_rec:9.4f} {t_prec:9.4f} {t_f1:9.4f}")

        print(sep)

print_summary(all_results)


# ══════════════════════════════════════════════════════════════
#  14. GROUPED BAR CHART — all models, both conditions
# ══════════════════════════════════════════════════════════════
def plot_comparison(all_results, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)

    model_names = list(all_results.keys())
    metrics     = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    conditions  = ['imbalanced', 'balanced']
    colors      = {'imbalanced': '#E74C3C', 'balanced': '#2ECC71'}

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flat

    for ax, metric_name in zip(axes, metrics):
        m_idx = metrics.index(metric_name)
        x     = np.arange(len(model_names))
        w     = 0.35

        for i, cond in enumerate(conditions):
            vals = [all_results[n][cond]['test'][m_idx] for n in model_names]
            bars = ax.bar(x + i*w, vals, w,
                          label=cond.capitalize(),
                          color=colors[cond], alpha=0.85,
                          edgecolor='white', linewidth=0.8)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        v + 0.005, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=7.5)

        ax.set_title(metric_name, fontsize=13, fontweight='bold')
        ax.set_xticks(x + w/2)
        ax.set_xticklabels(model_names, rotation=18, ha='right', fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel('Score')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle('Model Comparison — Imbalanced vs Balanced\n(One-vs-All: Digit 0)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\nComparison chart saved → {path}")

plot_comparison(all_results)


# ══════════════════════════════════════════════════════════════
#  15. SIDE-BY-SIDE CONFUSION MATRICES  (all models, one figure)
# ══════════════════════════════════════════════════════════════
def plot_all_cms(all_results, save_dir='plots'):
    """
    One 2-row × 5-column figure.
    Top row    = Imbalanced.
    Bottom row = Balanced.
    """
    os.makedirs(save_dir, exist_ok=True)
    model_names = list(all_results.keys())
    n = len(model_names)

    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))

    for col, name in enumerate(model_names):
        for row, cond in enumerate(['imbalanced', 'balanced']):
            ax             = axes[row][col]
            tp, fn, fp, tn = all_results[name][cond]['cm']
            cm = np.array([[tp, fn], [fp, tn]])
            total = cm.sum()
            annot = np.array([
                [f"{tp}\n{tp/total*100:.1f}%", f"{fn}\n{fn/total*100:.1f}%"],
                [f"{fp}\n{fp/total*100:.1f}%", f"{tn}\n{tn/total*100:.1f}%"]
            ])
            sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax,
                        xticklabels=['Pred 0', 'Pred Other'],
                        yticklabels=['True 0', 'True Other'],
                        linewidths=0.8, linecolor='white',
                        cbar=False)
            f1 = all_results[name][cond]['test'][3]
            ax.set_title(f"{name}\n[{cond.capitalize()}]  F1={f1:.3f}",
                         fontsize=9)
            if col == 0:
                ax.set_ylabel(cond.capitalize(), fontsize=9, fontweight='bold')

    plt.suptitle('Confusion Matrices — All Models × Both Conditions',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'all_confusion_matrices.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"All-CM figure saved → {path}")

plot_all_cms(all_results)
