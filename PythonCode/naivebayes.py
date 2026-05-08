import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from skopt.space import Real, Integer
from skopt import gp_minimize


PCA_COMPONENTS = 100
SMOOTHING      = 1e-6


# HOG settings 
HOG_CELL_SIZE  = 4   
HOG_BLOCK_SIZE = 2   
HOG_BINS       = 9   


BAYESIAN_SEARCH_ITER = 30
PCA_MIN              = 50
PCA_MAX              = 120
SMOOTHING_LOG_LOW    = 1e-9
SMOOTHING_LOG_HIGH   = 1e-1

# HOG search ranges 
HOG_CELL_SIZE_MIN  = 3
HOG_CELL_SIZE_MAX  = 5
HOG_BLOCK_SIZE_MIN = 1
HOG_BLOCK_SIZE_MAX = 2
HOG_BINS_MIN       = 7
HOG_BINS_MAX       = 12

# Bias-variance analysis PCA range
BIAS_VARIANCE_PCA_GRID = [10, 30, 50, 75, 100, 150, 200]

# Boosting settings
BOOST_N_ESTIMATORS = 25


# DATA LOADING
base_path = r"G:/ImageClassification/dataset/"

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28 * 28) / 255.0

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)

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

    return X_train_pca, X_test_pca

# HOG FEATURE EXTRACTION 
def hog_features(images, cell_size=4, block_size=2, bins=9):
    # Accept both flat (n, 784) and shaped (n, 28, 28) input
    if images.ndim == 2:
        n    = images.shape[0]
        side = int(np.sqrt(images.shape[1]))
        images = images.reshape(n, side, side)

    images = images.astype(np.float32)
    n, h, w = images.shape
    eps = 1e-6

    # Step 1 — Gradients across all images at once (n, h, w)
    gx = np.zeros_like(images)
    gy = np.zeros_like(images)
    gx[:, :, 1:-1] = images[:, :, 2:] - images[:, :, :-2]
    gy[:, 1:-1, :] = images[:, 2:, :] - images[:, :-2, :]

    # Step 2 — Magnitude and orientation for all images (n, h, w)
    magnitude   = np.sqrt(gx**2 + gy**2)
    orientation = (np.arctan2(gy, gx) * 180 / np.pi) % 180

    n_cells_x = w // cell_size
    n_cells_y = h // cell_size

    # Step 3 — Build histogram tensor (n, n_cells_y, n_cells_x, bins)
    bin_idx = np.clip((orientation // (180 / bins)).astype(int), 0, bins - 1)

    hist_tensor = np.zeros((n, n_cells_y, n_cells_x, bins), dtype=np.float32)

    for b in range(bins):
        # Mask of pixels belonging to this bin — shape (n, h, w)
        mask = (bin_idx == b).astype(np.float32)
        weighted = magnitude * mask 

        # Sum contributions into cells using reshape + sum
        # Crop to cell-aligned region first
        weighted_cropped = weighted[:, :n_cells_y * cell_size, :n_cells_x * cell_size]
        # (n, n_cells_y, cell_size, n_cells_x, cell_size) → sum over cell dims
        hist_tensor[:, :, :, b] = weighted_cropped.reshape(
            n, n_cells_y, cell_size, n_cells_x, cell_size
        ).sum(axis=(2, 4))

    # Step 4 — Block normalisation across all images at once
    n_blocks_y = n_cells_y - block_size + 1
    n_blocks_x = n_cells_x - block_size + 1
    n_block_feats = block_size * block_size * bins

    hog_matrix = np.zeros((n, n_blocks_y * n_blocks_x * n_block_feats), dtype=np.float32)

    block_idx = 0
    for i in range(n_blocks_y):
        for j in range(n_blocks_x):
            # Extract block for all images: (n, block_size, block_size, bins)
            block = hist_tensor[:, i:i+block_size, j:j+block_size, :]
            block_flat = block.reshape(n, -1)  # (n, block_size*block_size*bins)

            # L2 normalise each block across all images
            norms = np.sqrt((block_flat**2).sum(axis=1, keepdims=True)) + eps
            block_flat = block_flat / norms

            start = block_idx * n_block_feats
            end   = start + n_block_feats
            hog_matrix[:, start:end] = block_flat
            block_idx += 1

    return hog_matrix


# FEATURE PIPELINE
def apply_pca(X_train, X_other, n_components):
    """Fit PCA on X_train only, apply to both. Called inside fold loops."""
    X_train_new, X_other_new = pca(X_train, X_other, n_components)
    return X_train_new, X_other_new


def precompute_hog(X, cell_size=None, block_size=None, bins=None):
    """
    Runs vectorized HOG on the full dataset.
    Safe outside fold loops — HOG fits no parameters.
    Falls back to global HOG settings if params not specified.
    """
    cs = cell_size  or HOG_CELL_SIZE
    bs = block_size or HOG_BLOCK_SIZE
    bn = bins       or HOG_BINS
    print(f"  Computing HOG (cell={cs}, block={bs}, bins={bn}) on {len(X)} images...")
    return hog_features(X, cs, bs, bn)



# NAIVE BAYES
class GaussianNB:
    def __init__(self, smoothing=1e-6):
        self.smoothing = smoothing
        self.classes = []
        self.priors = {}
        self.means = {}
        self.vars = {}

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)
        n_samples = len(X)

        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / n_samples
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0) + self.smoothing

    def predict(self, X):
        X = np.array(X)
        log_posteriors = []

        for c in self.classes:
            log_prior = np.log(self.priors[c])
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * self.vars[c])
                + ((X - self.means[c]) ** 2) / self.vars[c],
                axis=1
            )
            log_posteriors.append(log_prior + log_likelihood)

        return self.classes[np.argmax(np.vstack(log_posteriors), axis=0)]

    # AdaBoost compatibility

    @property
    def classes_(self):
        return self.classes

    def get_params(self, deep=True):
        return {"smoothing": self.smoothing}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y, sample_weight=None):
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)
        n_samples = len(X)

        if sample_weight is None:
            for c in self.classes:
                X_c = X[y == c]
                self.priors[c] = len(X_c) / n_samples
                self.means[c]  = np.mean(X_c, axis=0)
                self.vars[c]   = np.var(X_c, axis=0) + self.smoothing
        else:
            sample_weight = np.array(sample_weight, dtype=np.float64)
            sample_weight = sample_weight / sample_weight.sum()

            for c in self.classes:
                mask = y == c
                X_c  = X[mask]
                w_c  = sample_weight[mask]

                self.priors[c] = w_c.sum()
                self.means[c]  = np.average(X_c, axis=0, weights=w_c)
                self.vars[c]   = np.average(
                    (X_c - self.means[c]) ** 2, axis=0, weights=w_c
                ) + self.smoothing

        return self

    def predict_proba(self, X):
        X = np.array(X)
        log_posteriors = []

        for c in self.classes:
            log_prior = np.log(self.priors[c])
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * self.vars[c])
                + ((X - self.means[c]) ** 2) / self.vars[c],
                axis=1
            )
            log_posteriors.append(log_prior + log_likelihood)

        log_posteriors = np.vstack(log_posteriors).T
        log_posteriors -= log_posteriors.max(axis=1, keepdims=True)
        proba = np.exp(log_posteriors)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba


class AdaBoost:
    def __init__(self, n_estimators=50, smoothing=1e-6):
        self.n_estimators = n_estimators
        self.smoothing = smoothing

        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = len(X)
        n_classes = len(np.unique(y))

        # Initialize uniform sample weights
        weights = np.ones(n_samples) / n_samples

        for t in range(self.n_estimators):

            # Train weak learner
            model = GaussianNB(smoothing=self.smoothing)
            model.fit(X, y, sample_weight=weights)

            predictions = model.predict(X)

            # Weighted error
            incorrect = (predictions != y)
            error = np.sum(weights * incorrect)

            # Avoid divide-by-zero
            error = np.clip(error, 1e-10, 1 - 1e-10)

            # SAMME alpha
            alpha = np.log((1 - error) / error) + np.log(n_classes - 1)

            # Update sample weights
            weights *= np.exp(alpha * incorrect)

            # Normalize
            weights /= np.sum(weights)

            # Save learner
            self.models.append(model)
            self.alphas.append(alpha)

            print(
                f"  Estimator {t+1:>2}/{self.n_estimators} "
                f"Error={error:.4f} Alpha={alpha:.4f}"
            )

    def predict(self, X):
        classes = self.models[0].classes
        class_scores = np.zeros((len(X), len(classes)))

        for alpha, model in zip(self.alphas, self.models):
            predictions = model.predict(X)

            for i, c in enumerate(classes):
                class_scores[:, i] += alpha * (predictions == c)

        return classes[np.argmax(class_scores, axis=1)]



# EVALUATION
def evaluate(y_true, y_pred):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='macro'),
        "recall":    recall_score(y_true, y_pred, average='macro'),
        "f1":        f1_score(y_true, y_pred, average='macro')
    }

def print_results(name, y_true, y_pred):
    print("=" * 60)
    print(name)
    print("=" * 60)
    print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred, average='macro'), 4))
    print("Recall   :", round(recall_score(y_true, y_pred, average='macro'), 4))
    print("F1-Score :", round(f1_score(y_true, y_pred, average='macro'), 4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))



def _evaluate_combo(X_train, y_train, kf, n_components, smoothing):
    fold_accs = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        X_tr, X_val = apply_pca(X_tr, X_val, n_components)

        model = GaussianNB(smoothing=smoothing)
        model.fit(X_tr, y_tr)
        fold_accs.append(accuracy_score(y_val, model.predict(X_val)))

    return float(np.mean(fold_accs))


def _print_top5(all_results):
    top5 = sorted(all_results, key=lambda x: x[-1], reverse=True)[:5]
    print("\n  Top 5 combinations found:")
    print(f"  {'PCA':>5}  {'Smoothing':>12}  {'Cell':>5}  {'Block':>6}  {'Bins':>5}  {'CV Acc':>8}")
    print("  " + "-" * 52)
    for row in top5:
        pca, sm, cs, bs, bn, acc = row
        print(f"  {pca:>5}  {sm:>12.2e}  {cs:>5}  {bs:>6}  {bn:>5}  {acc:>8.4f}")




# Bayesian Hyperparameter Optimization
def run_hyperparameter_tuning(X_train, y_train):
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Bayesian Hyperparameter Optimization")
    print("=" * 60)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    call_count = [0]
    all_results = []
    best_so_far = [0.0, {}]

    print(f"\n  Searching: PCA + smoothing + HOG params")
    print(f"  Iterations  : {BAYESIAN_SEARCH_ITER}")
    print(f"  PCA range   : [{PCA_MIN}, {PCA_MAX}]")
    print(f"  Smoothing   : log-uniform [{SMOOTHING_LOG_LOW:.0e}, {SMOOTHING_LOG_HIGH:.0e}]")
    print(f"  cell_size   : [{HOG_CELL_SIZE_MIN}, {HOG_CELL_SIZE_MAX}]")
    print(f"  block_size  : [{HOG_BLOCK_SIZE_MIN}, {HOG_BLOCK_SIZE_MAX}]")
    print(f"  bins        : [{HOG_BINS_MIN}, {HOG_BINS_MAX}]")
    print(f"  Method      : Gaussian Process + Expected Improvement\n")

    search_space = [
        Integer(PCA_MIN, PCA_MAX, name='n_components'),
        Real(SMOOTHING_LOG_LOW, SMOOTHING_LOG_HIGH, prior='log-uniform', name='smoothing'),
        Integer(HOG_CELL_SIZE_MIN, HOG_CELL_SIZE_MAX, name='cell_size'),
        Integer(HOG_BLOCK_SIZE_MIN, HOG_BLOCK_SIZE_MAX, name='block_size'),
        Integer(HOG_BINS_MIN, HOG_BINS_MAX, name='bins'),
    ]

    def objective(params):
        n_components = int(params[0])
        smoothing = float(params[1])
        cell_size = int(params[2])
        block_size = int(params[3])
        bins = int(params[4])

        X_hog = hog_features(X_train, cell_size, block_size, bins)

        mean_acc = _evaluate_combo(
            X_hog,
            y_train,
            kf,
            n_components,
            smoothing
        )

        all_results.append((
            n_components,
            smoothing,
            cell_size,
            block_size,
            bins,
            mean_acc
        ))

        call_count[0] += 1

        marker = ""
        if mean_acc > best_so_far[0]:
            best_so_far[0] = mean_acc
            best_so_far[1] = {
                "n_components": n_components,
                "smoothing": smoothing,
                "cell_size": cell_size,
                "block_size": block_size,
                "bins": bins
            }
            marker = "  ◄ best"

        print(
            f"  [{call_count[0]:>2}/{BAYESIAN_SEARCH_ITER}]  "
            f"PCA={n_components:>3}  sm={smoothing:.1e}  "
            f"cell={cell_size}  blk={block_size}  bins={bins}  "
            f"CV={mean_acc:.4f}{marker}"
        )

        return -mean_acc

    gp_minimize(
        objective,
        search_space,
        n_calls=BAYESIAN_SEARCH_ITER,
        random_state=42,
        verbose=False
    )

    best_params = best_so_far[1]
    print(f"\n  Best params : {best_params}")
    print(f"  Best CV Acc : {best_so_far[0]:.4f}")

    _print_top5(all_results)

    return best_params



# Bias-Variance Analysis
def run_bias_variance_analysis(X_train, y_train):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Bias-Variance Analysis")
    print("=" * 60)

    X_train = precompute_hog(X_train)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_accs, val_accs = [], []

    for n_components in BIAS_VARIANCE_PCA_GRID:
        fold_train_accs, fold_val_accs = [], []

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            X_tr, X_val = apply_pca(X_tr, X_val, n_components)

            model = GaussianNB(smoothing=SMOOTHING)
            model.fit(X_tr, y_tr)

            fold_train_accs.append(accuracy_score(y_tr,  model.predict(X_tr)))
            fold_val_accs.append(accuracy_score(y_val, model.predict(X_val)))

        train_accs.append(np.mean(fold_train_accs))
        val_accs.append(np.mean(fold_val_accs))
        gap = train_accs[-1] - val_accs[-1]
        print(f"  PCA={n_components:>3}  Train={train_accs[-1]:.4f}  "
              f"Val={val_accs[-1]:.4f}  Gap={gap:.4f}")

    plt.figure(figsize=(9, 5))
    plt.plot(BIAS_VARIANCE_PCA_GRID, train_accs, marker='o', label='Train Accuracy')
    plt.plot(BIAS_VARIANCE_PCA_GRID, val_accs,   marker='o', label='Validation Accuracy')
    plt.fill_between(BIAS_VARIANCE_PCA_GRID, train_accs, val_accs,
                     alpha=0.1, label='Bias-Variance Gap')
    plt.xlabel('PCA Components')
    plt.ylabel('Accuracy')
    plt.title(f'Bias-Variance Analysis: Train vs Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Boosting
def run_boosting(X_train_feat, y_train, X_test_feat, y_test):
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: AdaBoost vs Baseline GNB")
    print("=" * 60)

    # 1. Baseline Gaussian NB
    baseline = GaussianNB(smoothing=SMOOTHING)
    baseline.fit(X_train_feat, y_train)
    baseline_pred = baseline.predict(X_test_feat)
    print_results("Baseline GNB", y_test, baseline_pred)

    # 2. AdaBoost
    boosted = AdaBoost(
        n_estimators=BOOST_N_ESTIMATORS,
        smoothing=SMOOTHING
    )
    boosted.fit(X_train_feat, y_train)
    boosted_pred = boosted.predict(X_test_feat)
    print_results("AdaBoost + GNB", y_test, boosted_pred)

    # 3. Calculate Difference
    diff = accuracy_score(y_test, boosted_pred) - accuracy_score(y_test, baseline_pred)
    print(f"Accuracy Difference : {diff:+.4f}")

  
    # PLOTTING 
  
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    baseline_vals = [
        accuracy_score(y_test, baseline_pred),
        precision_score(y_test, baseline_pred, average='macro'),
        recall_score(y_test, baseline_pred,    average='macro'),
        f1_score(y_test, baseline_pred,        average='macro'),
    ]
    
    boosted_vals = [
        accuracy_score(y_test, boosted_pred),
        precision_score(y_test, boosted_pred, average='macro'),
        recall_score(y_test, boosted_pred,    average='macro'),
        f1_score(y_test, boosted_pred,        average='macro'),
    ]

    x     = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline GNB',       color='#4C72B0')
    bars2 = ax.bar(x + width/2, boosted_vals,  width, label='AdaBoost+GNB', color='#C44E52')

    # Add text labels on top of the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(bars1)
    autolabel(bars2)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylabel('Score')
    ax.set_title('Comparison: Baseline GNB vs AdaBoost + GNB')
    ax.legend(loc='lower right')
    
    # Adjust Y-axis to zoom in on the differences
    min_val = min(baseline_vals + boosted_vals)
    ax.set_ylim(max(0, min_val - 0.05), 1.05)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.show()

    # Confusion Matrices 
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, baseline_pred),
        display_labels=range(10)
    ).plot(ax=axes[0], colorbar=True, cmap='Blues')
    axes[0].set_title(
        f'Baseline GNB  —  Acc: {accuracy_score(y_test, baseline_pred):.4f}',
        fontsize=12, pad=10
    )

    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, boosted_pred),
        display_labels=range(10)
    ).plot(ax=axes[1], colorbar=True, cmap='Reds')
    axes[1].set_title(
        f'AdaBoost + GNB  —  Acc: {accuracy_score(y_test, boosted_pred):.4f}',
        fontsize=12, pad=10
    )

    plt.suptitle('Confusion Matrix Comparison: Baseline GNB vs AdaBoost + GNB',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()



# MAIN
def main():
    print("Loading data...")

    x_train = load_mnist_images(base_path + 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    y_train = load_mnist_labels(base_path + 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    x_test  = load_mnist_images(base_path + 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    y_test  = load_mnist_labels(base_path + 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    X_train_full, y_train_full = x_train, y_train
    X_test, y_test = x_test, y_test

    print("Train samples:", len(X_train_full))
    print("Test samples :", len(X_test))


    # Bayesian Hyperparameter Tuning
    n_components = PCA_COMPONENTS
    smoothing    = SMOOTHING
    cell_size    = HOG_CELL_SIZE
    block_size   = HOG_BLOCK_SIZE
    bins         = HOG_BINS

    best = run_hyperparameter_tuning(X_train_full, y_train_full)
    n_components = best.get("n_components", PCA_COMPONENTS)
    smoothing    = best.get("smoothing",    SMOOTHING)
    cell_size    = best.get("cell_size",    HOG_CELL_SIZE)
    block_size   = best.get("block_size",   HOG_BLOCK_SIZE)
    bins         = best.get("bins",         HOG_BINS)
    print(f"\nUsing tuned params: PCA={n_components}, smoothing={smoothing:.2e}", end="")
    print(f", cell={cell_size}, block={block_size}, bins={bins}", end="")
    print()

    
    # Bias-Variance Analysis
   
    run_bias_variance_analysis(X_train_full, y_train_full)

 
    X_train_pipeline = X_train_full
    X_test_pipeline  = X_test
    print("\nPrecomputing HOG for main pipeline...")
    X_train_pipeline = precompute_hog(X_train_full, cell_size, block_size, bins)
    X_test_pipeline  = precompute_hog(X_test,       cell_size, block_size, bins)

    
    # K-FOLD CROSS VALIDATION
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    print("\nRunning K-Fold Cross Validation...\n")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_pipeline)):
        X_tr  = X_train_pipeline[train_idx]
        y_tr  = y_train_full[train_idx]
        X_val = X_train_pipeline[val_idx]
        y_val = y_train_full[val_idx]

        X_tr, X_val = apply_pca(X_tr, X_val, n_components)

        model = GaussianNB(smoothing=smoothing)
        model.fit(X_tr, y_tr)

        y_pred  = model.predict(X_val)
        metrics = evaluate(y_val, y_pred)
        results.append(metrics)

        print(f"Fold {fold+1} Accuracy:", round(metrics["accuracy"], 4))

    print(f"\nCV Average Accuracy : {np.mean([r['accuracy'] for r in results]):.4f}")
    print(f"CV Std Dev          : {np.std([r['accuracy']  for r in results]):.4f}")

   
    # FINAL TRAINING ON FULL DATA
    print("\nTraining final model...")

    X_train_feat = X_train_pipeline
    X_test_feat  = X_test_pipeline

    X_train_feat, X_test_feat = apply_pca(X_train_pipeline, X_test_pipeline, n_components)

    final_model = GaussianNB(smoothing=smoothing)
    final_model.fit(X_train_feat, y_train_full)

    y_test_pred = final_model.predict(X_test_feat)
    print_results("Final Test Results", y_test, y_test_pred)

    # Boosting
    run_boosting(X_train_feat, y_train_full, X_test_feat, y_test)


if __name__ == "__main__":
    main()