import os
import numpy as np
import pandas as pd

from itertools import product as iterproduct

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from skimage.feature import hog

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# DATA LOADING
# =========================================================

def _stratified_sample(y, n, rng):
    classes = np.unique(y)
    per_class = n // len(classes)

    idx = []
    for c in classes:
        class_idx = np.where(y == c)[0]
        chosen = rng.choice(class_idx,
                            min(per_class, len(class_idx)),
                            replace=False)
        idx.append(chosen)

    return np.concatenate(idx)


def load_data(n_train=None, n_test=None):
    print("[Data] Loading MNIST...")

    train_df = pd.read_csv(
        '/content/sample_data/mnist_train_small.csv',
        header=None
    )

    test_df = pd.read_csv(
        '/content/sample_data/mnist_test.csv',
        header=None
    )

    y_train_raw = train_df.iloc[:, 0].values.astype(int)
    X_train_raw = train_df.iloc[:, 1:].values.astype(np.float32) / 255.0

    y_test_raw = test_df.iloc[:, 0].values.astype(int)
    X_test_raw = test_df.iloc[:, 1:].values.astype(np.float32) / 255.0

    rng = np.random.default_rng(42)

    if n_train is not None:
        tr_idx = _stratified_sample(y_train_raw, n_train, rng)
        X_train = X_train_raw[tr_idx]
        y_train = y_train_raw[tr_idx]
    else:
        X_train = X_train_raw
        y_train = y_train_raw

    if n_test is not None:
        te_idx = _stratified_sample(y_test_raw, n_test, rng)
        X_test = X_test_raw[te_idx]
        y_test = y_test_raw[te_idx]
    else:
        X_test = X_test_raw
        y_test = y_test_raw

    idx = rng.permutation(len(X_train))
    split = int(0.85 * len(X_train))

    X_tr = X_train[idx[:split]]
    y_tr = y_train[idx[:split]]

    X_val = X_train[idx[split:]]
    y_val = y_train[idx[split:]]

    print(f"[Data] Train={len(X_tr)} | Val={len(X_val)} | Test={len(X_test)}")

    return X_tr, y_tr, X_val, y_val, X_test, y_test


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_hog_features(X):
    return np.array([
        hog(
            img.reshape(28, 28),
            pixels_per_cell=(7, 7),
            cells_per_block=(2, 2),
            feature_vector=True
        )
        for img in X
    ])


def extract_features(X_train, X_val, X_test, n_pca=50):
    print("[Features] Extracting HOG...")

    hog_tr = extract_hog_features(X_train)
    hog_val = extract_hog_features(X_val)
    hog_te = extract_hog_features(X_test)

    print(f"[Features] PCA ({n_pca})...")

    pca = PCA(n_components=n_pca, random_state=42)

    pca_tr = pca.fit_transform(X_train)
    pca_val = pca.transform(X_val)
    pca_te = pca.transform(X_test)

    print(f"[Features] Explained Variance = "
          f"{pca.explained_variance_ratio_.sum():.3f}")

    X_tr_f = np.hstack([X_train, pca_tr, hog_tr])
    X_val_f = np.hstack([X_val, pca_val, hog_val])
    X_te_f = np.hstack([X_test, pca_te, hog_te])

    scaler = StandardScaler()

    X_tr_f = scaler.fit_transform(X_tr_f)
    X_val_f = scaler.transform(X_val_f)
    X_te_f = scaler.transform(X_te_f)

    print(f"[Features] Dimension = {X_tr_f.shape[1]}")

    return X_tr_f, X_val_f, X_te_f


# =========================================================
# KERNELS
# =========================================================

def linear_kernel(X, Z):
    return X @ Z.T


def rbf_kernel(X, Z, gamma=0.01):
    sq_x = np.sum(X ** 2, axis=1, keepdims=True)
    sq_z = np.sum(Z ** 2, axis=1, keepdims=True)

    dist2 = sq_x + sq_z.T - 2.0 * (X @ Z.T)

    return np.exp(-gamma * dist2)


def poly_kernel(X, Z, degree=3, coef0=1.0):
    return (X @ Z.T + coef0) ** degree


# =========================================================
# BINARY KERNEL SVM
# =========================================================

class KernelSVM:

    def __init__(
        self,
        C=1.0,
        kernel='rbf',
        gamma=0.01,
        degree=3,
        max_iter=100,
        tol=1e-3
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol

        self.convergence_curve_ = []

    def _K(self, X, Z):

        if self.kernel == 'rbf':
            return rbf_kernel(X, Z, self.gamma)

        if self.kernel == 'linear':
            return linear_kernel(X, Z)

        if self.kernel == 'poly':
            return poly_kernel(X, Z, self.degree)

        raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X, y):

        y_ = np.where(y == 0, -1.0, 1.0).astype(float)

        n = X.shape[0]

        K = self._K(X, X)

        alpha = np.zeros(n)
        b = 0.0

        rng = np.random.default_rng(42)

        self.convergence_curve_ = []

        for _ in range(self.max_iter):

            changed = 0

            for i in range(n):

                Ei = float(np.dot(alpha * y_, K[i]) + b) - y_[i]

                condition = (
                    (y_[i] * Ei < -self.tol and alpha[i] < self.C)
                    or
                    (y_[i] * Ei > self.tol and alpha[i] > 0)
                )

                if not condition:
                    continue

                j = rng.integers(0, n)

                while j == i:
                    j = rng.integers(0, n)

                Ej = float(np.dot(alpha * y_, K[j]) + b) - y_[j]

                ai_old = alpha[i]
                aj_old = alpha[j]

                if y_[i] != y_[j]:
                    L = max(0.0, alpha[j] - alpha[i])
                    H = min(self.C, self.C + alpha[j] - alpha[i])
                else:
                    L = max(0.0, alpha[i] + alpha[j] - self.C)
                    H = min(self.C, alpha[i] + alpha[j])

                if L >= H:
                    continue

                eta = 2.0 * K[i, j] - K[i, i] - K[j, j]

                if eta >= 0:
                    continue

                alpha[j] -= y_[j] * (Ei - Ej) / eta
                alpha[j] = np.clip(alpha[j], L, H)

                if abs(alpha[j] - aj_old) < 1e-5:
                    continue

                alpha[i] += y_[i] * y_[j] * (aj_old - alpha[j])

                b1 = (
                    b - Ei
                    - y_[i] * (alpha[i] - ai_old) * K[i, i]
                    - y_[j] * (alpha[j] - aj_old) * K[i, j]
                )

                b2 = (
                    b - Ej
                    - y_[i] * (alpha[i] - ai_old) * K[i, j]
                    - y_[j] * (alpha[j] - aj_old) * K[j, j]
                )

                if 0 < alpha[i] < self.C:
                    b = b1
                elif 0 < alpha[j] < self.C:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                changed += 1

            self.convergence_curve_.append(changed)

            if changed == 0:
                break

        sv_mask = alpha > 1e-5

        self.sv_alpha = alpha[sv_mask]
        self.sv_y = y_[sv_mask]
        self.sv_X = X[sv_mask]

        self.alpha = alpha
        self.b = b

    def decision_function(self, X):

        K = self._K(self.sv_X, X)

        return (self.sv_alpha * self.sv_y) @ K + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, 0)


# =========================================================
# ONE VS REST
# =========================================================

class OvRKernelSVM:

    def __init__(
        self,
        C=1.0,
        kernel='rbf',
        gamma=0.01,
        degree=3,
        max_iter=100,
        tol=1e-3
    ):
        self.params = dict(
            C=C,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            max_iter=max_iter,
            tol=tol
        )

        self.classifiers = {}
        self.classes_ = None

    def fit(self, X, y):

        self.classes_ = np.unique(y)

        for c in self.classes_:
            print(f"[OvR] Training digit {c}...", end='\r')

            clf = KernelSVM(**self.params)

            clf.fit(X, (y == c).astype(int))

            self.classifiers[c] = clf

        print()

    def decision_scores(self, X):

        return np.column_stack([
            self.classifiers[c].decision_function(X)
            for c in self.classes_
        ])

    def predict(self, X):

        scores = self.decision_scores(X)

        return self.classes_[np.argmax(scores, axis=1)]


# =========================================================
# CROSS VALIDATION
# =========================================================

def k_fold_cv_score(X, y, k=3, **svm_params):

    n = len(X)
    fold_size = n // k

    accs = []

    for fold in range(k):

        val_idx = np.arange(
            fold * fold_size,
            (fold + 1) * fold_size
        )

        train_idx = np.concatenate([
            np.arange(0, fold * fold_size),
            np.arange((fold + 1) * fold_size, n)
        ])

        clf = OvRKernelSVM(**svm_params)

        clf.fit(X[train_idx], y[train_idx])

        pred = clf.predict(X[val_idx])

        acc = accuracy_score(y[val_idx], pred)

        accs.append(acc)

        print(f"Fold {fold + 1}/{k} | Acc = {acc:.4f}")

    return float(np.mean(accs))


def hyperparameter_search(X_tr, y_tr, k=3):

    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'gamma': [0.001, 0.005, 0.01]
    }

    best_score = -1
    best_params = {}

    print("\n[Search] Grid Search")

    for C, gamma in iterproduct(
        param_grid['C'],
        param_grid['gamma']
    ):

        print(f"C={C} | gamma={gamma}")

        score = k_fold_cv_score(
            X_tr,
            y_tr,
            k=k,
            C=C,
            kernel='rbf',
            gamma=gamma,
            max_iter=50,
            tol=1e-3
        )

        print(f"CV Acc = {score:.4f}")

        if score > best_score:
            best_score = score
            best_params = dict(C=C, gamma=gamma)

    print(f"\nBest Params = {best_params}")
    print(f"Best CV Acc = {best_score:.4f}")

    return best_params


# =========================================================
# BIAS VARIANCE
# =========================================================

def bias_variance_analysis(
    X_tr,
    y_tr,
    X_val,
    y_val,
    gamma=0.01,
    C_values=None,
    save_dir="/content"
):

    if C_values is None:
        C_values = [0.01, 0.1, 1.0, 10.0, 50.0]

    train_accs = []
    val_accs = []

    print("\n[Bias-Variance]")

    for C in C_values:

        clf = OvRKernelSVM(
            C=C,
            kernel='rbf',
            gamma=gamma,
            max_iter=50,
            tol=1e-3
        )

        clf.fit(X_tr, y_tr)

        tr_acc = accuracy_score(y_tr, clf.predict(X_tr))
        val_acc = accuracy_score(y_val, clf.predict(X_val))

        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        gap = tr_acc - val_acc

        if gap > 0.05:
            state = "Overfitting"
        elif val_acc < 0.70:
            state = "Underfitting"
        else:
            state = "Good"

        print(
            f"C={C:<5} "
            f"Train={tr_acc:.4f} "
            f"Val={val_acc:.4f} "
            f"Gap={gap:.4f} "
            f"{state}"
        )

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.semilogx(C_values, train_accs, 'o-', label='Train')
    ax.semilogx(C_values, val_accs, 's--', label='Validation')

    ax.set_xlabel("C")
    ax.set_ylabel("Accuracy")
    ax.set_title("Bias-Variance")

    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    path = os.path.join(save_dir, "bias_variance_analysis.png")

    plt.savefig(path, dpi=150)

    plt.show()
    plt.close(fig)


# =========================================================
# BOOSTED SVM
# =========================================================

class BoostedKernelSVM:

    def __init__(self, n_estimators=5, **svm_params):

        self.n_estimators = n_estimators
        self.svm_params = svm_params

        self.estimators_ = []
        self.alphas_ = []
        self.train_errors_ = []

    def fit(self, X, y):

        n = len(X)

        w = np.ones(n) / n

        rng = np.random.default_rng(0)

        for t in range(self.n_estimators):

            print(f"\n[Boost] Round {t + 1}/{self.n_estimators}")

            idx = rng.choice(n, size=n, replace=True, p=w)

            clf = OvRKernelSVM(**self.svm_params)

            clf.fit(X[idx], y[idx])

            correct = (clf.predict(X) == y).astype(float)

            err = np.dot(w, 1.0 - correct)

            err = np.clip(err, 1e-10, 1 - 1e-10)

            alpha = 0.5 * np.log((1.0 - err) / err)

            w *= np.exp(-alpha * (2 * correct - 1))
            w /= w.sum()

            self.estimators_.append(clf)
            self.alphas_.append(alpha)
            self.train_errors_.append(err)

            print(f"Error={err:.4f} | Alpha={alpha:.4f}")

        return self

    def predict(self, X):

        classes = self.estimators_[0].classes_

        combined = sum(
            a * clf.decision_scores(X)
            for clf, a in zip(self.estimators_, self.alphas_)
        )

        return classes[np.argmax(combined, axis=1)]


# =========================================================
# EVALUATION
# =========================================================

def evaluate_multiclass(
    model,
    X,
    y,
    name="",
    save_dir="/content"
):

    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)

    prec = precision_score(
        y,
        y_pred,
        average='macro',
        zero_division=0
    )

    rec = recall_score(
        y,
        y_pred,
        average='macro',
        zero_division=0
    )

    f1 = f1_score(
        y,
        y_pred,
        average='macro',
        zero_division=0
    )

    cm = confusion_matrix(y, y_pred)

    report = classification_report(
        y,
        y_pred,
        labels=list(range(10)),
        target_names=[f"Digit {i}" for i in range(10)],
        zero_division=0
    )

    print("\n" + "=" * 60)
    print(f"Report - {name}")
    print("=" * 60)

    print(report)

    print(
        f"Accuracy={acc:.4f} | "
        f"Precision={prec:.4f} | "
        f"Recall={rec:.4f} | "
        f"F1={f1:.4f}"
    )

    report_path = os.path.join(
        save_dir,
        f"report_{name.lower().replace(' ', '_')}.txt"
    )

    with open(report_path, 'w') as f:
        f.write(report)

    fig, ax = plt.subplots(figsize=(9, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax,
        xticklabels=[str(i) for i in range(10)],
        yticklabels=[str(i) for i in range(10)]
    )

    for i in range(10):
        ax.add_patch(
            plt.Rectangle(
                (i, i),
                1,
                1,
                fill=False,
                edgecolor='green',
                lw=2
            )
        )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_title(
        f"{name}\n"
        f"Accuracy={acc:.4f} | F1={f1:.4f}"
    )

    plt.tight_layout()

    cm_path = os.path.join(
        save_dir,
        f"cm_{name.lower().replace(' ', '_')}.png"
    )

    plt.savefig(cm_path, dpi=150)

    plt.show()
    plt.close(fig)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }


# =========================================================
# PLOTS
# =========================================================

def plot_smo_convergence(curves, labels, save_dir="/content"):

    fig, ax = plt.subplots(figsize=(8, 4))

    for curve, label in zip(curves, labels):
        ax.plot(
            range(1, len(curve) + 1),
            curve,
            'o-',
            linewidth=2,
            label=label
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Alpha Updates")

    ax.set_title("SMO Convergence")

    ax.legend()

    ax.grid(alpha=0.3)

    plt.tight_layout()

    path = os.path.join(save_dir, "smo_convergence.png")

    plt.savefig(path, dpi=150)

    plt.show()
    plt.close(fig)


def plot_boost_errors(errors, save_dir="/content"):

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        range(1, len(errors) + 1),
        errors,
        'o-',
        linewidth=2
    )

    ax.set_xlabel("Boost Round")
    ax.set_ylabel("Weighted Error")

    ax.set_title("Boosting Error")

    ax.grid(alpha=0.3)

    plt.tight_layout()

    path = os.path.join(save_dir, "boost_error_curve.png")

    plt.savefig(path, dpi=150)

    plt.show()
    plt.close(fig)


def plot_comparison(results_base, results_boost, save_dir="/content"):

    metrics = ['accuracy', 'precision', 'recall', 'f1']

    base_val = [results_base['Val (Base)'][m] for m in metrics]
    base_test = [results_base['Test (Base)'][m] for m in metrics]

    boost_val = [results_boost['Val (Boost)'][m] for m in metrics]
    boost_test = [results_boost['Test (Boost)'][m] for m in metrics]

    x = np.arange(len(metrics))

    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, title, base, boost in zip(
        axes,
        ['Validation', 'Test'],
        [base_val, base_test],
        [boost_val, boost_test]
    ):

        ax.bar(x - width / 2, base, width, label='Base')
        ax.bar(x + width / 2, boost, width, label='Boost')

        ax.set_xticks(x)

        ax.set_xticklabels(
            [m.capitalize() for m in metrics]
        )

        ax.set_ylim(0, 1.05)

        ax.set_title(title)

        ax.legend()

        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    path = os.path.join(save_dir, "baseline_vs_boosted.png")

    plt.savefig(path, dpi=150)

    plt.show()
    plt.close(fig)


def print_results_table(results):

    print("\n" + "=" * 62)

    print(
        f"{'Split':<18}"
        f"{'Accuracy':>10}"
        f"{'Precision':>10}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
    )

    print("-" * 62)

    for name, m in results.items():

        print(
            f"{name:<18}"
            f"{m['accuracy']:>10.4f}"
            f"{m['precision']:>10.4f}"
            f"{m['recall']:>10.4f}"
            f"{m['f1']:>10.4f}"
        )

    print("=" * 62)


# =========================================================
# MAIN
# =========================================================

SAVE_DIR = "/content"

np.random.seed(42)

# Load data
X_train, y_train, X_val, y_val, X_test, y_test = load_data()

# Feature extraction
X_train, X_val, X_test = extract_features(
    X_train,
    X_val,
    X_test,
    n_pca=50
)

# Bias-variance analysis
bv_idx = np.random.default_rng(1).choice(
    len(X_train),
    800,
    replace=False
)

bias_variance_analysis(
    X_train[bv_idx],
    y_train[bv_idx],
    X_val,
    y_val,
    gamma=0.01,
    C_values=[0.01, 0.1, 1.0, 10.0, 50.0],
    save_dir=SAVE_DIR
)

# Hyperparameter search
hp_idx = np.random.default_rng(2).choice(
    len(X_train),
    800,
    replace=False
)

best_params = hyperparameter_search(
    X_train[hp_idx],
    y_train[hp_idx],
    k=3
)

final_svm_params = dict(
    C=best_params['C'],
    kernel='rbf',
    gamma=best_params['gamma'],
    max_iter=100,
    tol=1e-3
)

# Baseline SVM
print("\n[Train] Baseline OvR SVM")

base_model = OvRKernelSVM(**final_svm_params)

base_model.fit(X_train, y_train)

conv_curve = base_model.classifiers[0].convergence_curve_

plot_smo_convergence(
    [conv_curve],
    ["Digit 0"],
    SAVE_DIR
)

results_base = {}

results_base['Val (Base)'] = evaluate_multiclass(
    base_model,
    X_val,
    y_val,
    "Val Base",
    SAVE_DIR
)

results_base['Test (Base)'] = evaluate_multiclass(
    base_model,
    X_test,
    y_test,
    "Test Base",
    SAVE_DIR
)

# Boosted SVM
print("\n[Train] Boosted SVM")

boosted_model = BoostedKernelSVM(
    n_estimators=3,
    **final_svm_params
)

boosted_model.fit(X_train, y_train)

plot_boost_errors(
    boosted_model.train_errors_,
    SAVE_DIR
)

results_boost = {}

results_boost['Val (Boost)'] = evaluate_multiclass(
    boosted_model,
    X_val,
    y_val,
    "Val Boost",
    SAVE_DIR
)

results_boost['Test (Boost)'] = evaluate_multiclass(
    boosted_model,
    X_test,
    y_test,
    "Test Boost",
    SAVE_DIR
)

# Summary
all_results = {
    **results_base,
    **results_boost
}

print_results_table(all_results)

plot_comparison(
    results_base,
    results_boost,
    SAVE_DIR
)

print("\n[Done] Saved to /content")
