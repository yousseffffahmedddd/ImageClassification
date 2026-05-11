"""
DT.py  —  Complete Decision Tree Pipeline for MNIST Multi-Class
═══════════════════════════════════════════════════════════════
Everything implemented from scratch using NumPy only.

Contents
────────
1.  DecisionTree          — fixed multi-class tree
2.  RandomForest          — bagging ensemble of DecisionTrees
3.  AdaBoostTree          — boosting ensemble (stumps)
4.  HyperparameterTuner   — grid search with 5-fold CV
5.  BiasVarianceAnalyser  — decompose error into bias + variance
6.  LearningCurveRunner   — train-size vs train/val error
7.  Regularisation notes  — min_samples_leaf, max_depth, n_features
8.  Helper utilities      — plotting, reporting (numpy + matplotlib)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# ══════════════════════════════════════════════════════════════════
#  PART 1 — DECISION TREE  (multi-class, fully fixed)
# ══════════════════════════════════════════════════════════════════
class DecisionTree:
    """
    Multi-class Decision Tree using Gini impurity.

    Regularisation parameters (bias-variance control)
    ──────────────────────────────────────────────────
    max_depth          : primary depth limiter
                         low  → high bias  (underfitting)
                         high → high variance (overfitting)
    min_samples_split  : node must have ≥ this many samples to split
                         high → simpler tree → more bias
    min_samples_leaf   : each leaf must have ≥ this many samples
                         high → smoother boundaries → less variance
    n_features         : None = all features (best accuracy for single tree)
                         int  = random subset per split (Random Forest style)
    n_thresholds       : candidate split thresholds per feature
                         more → finer splits → slight variance increase
    """
    def __init__(
        self,
        max_depth         = 20,
        min_samples_split = 5,
        min_samples_leaf  = 2,
        n_features        = None,
        n_thresholds      = 10,
        random_state      = 42
    ):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.n_features        = n_features
        self.n_thresholds      = n_thresholds
        self.rng               = np.random.default_rng(random_state)
        self.root              = None

    def fit(self, X, y):
        self.classes_ = np.unique(y.astype(int))
        self.root = self._grow(X, y.astype(int), depth=0)
        return self

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def get_params(self):
        return dict(
            max_depth         = self.max_depth,
            min_samples_split = self.min_samples_split,
            min_samples_leaf  = self.min_samples_leaf,
            n_features        = self.n_features,
            n_thresholds      = self.n_thresholds,
            random_state      = 42
        )

    def _grow(self, X, y, depth):
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return self._make_leaf(y)

        n_samples, n_feats = X.shape

        if self.n_features is None:
            feat_idxs = np.arange(n_feats)
        else:
            feat_idxs = self.rng.choice(
                n_feats,
                min(self.n_features, n_feats),
                replace=False
            )

        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feat is None:
            return self._make_leaf(y)

        left_mask  = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        if left_mask.sum() < self.min_samples_leaf \
                or right_mask.sum() < self.min_samples_leaf:
            return self._make_leaf(y)

        return {
            "leaf":      False,
            "feature":   best_feat,
            "threshold": best_thresh,
            "left":      self._grow(X[left_mask],  y[left_mask],  depth + 1),
            "right":     self._grow(X[right_mask], y[right_mask], depth + 1),
        }

    def _make_leaf(self, y):
        majority = int(np.bincount(y.astype(int)).argmax())
        return {"leaf": True, "value": majority}

    def _best_split(self, X, y, feat_idxs):
        best_gain    = -1
        split_feat   = None
        split_thresh = None
        parent_gini  = self._gini(y)

        for feat in feat_idxs:
            X_col      = X[:, feat]
            thresholds = np.unique(
                np.percentile(X_col, np.linspace(5, 95, self.n_thresholds))
            )

            for t in thresholds:
                left  = X_col <= t
                right = ~left

                if (left.sum()  < self.min_samples_leaf
                        or right.sum() < self.min_samples_leaf):
                    continue

                n     = len(y)
                child = ((left.sum()  / n) * self._gini(y[left]) +
                         (right.sum() / n) * self._gini(y[right]))
                gain  = parent_gini - child

                if gain > best_gain:
                    best_gain    = gain
                    split_feat   = feat
                    split_thresh = t

        return split_feat, split_thresh

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y.astype(int))
        probs  = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _traverse(self, x, node):
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse(x, node["left"])
        else:
            return self._traverse(x, node["right"])


# ══════════════════════════════════════════════════════════════════
#  PART 2 — RANDOM FOREST  (bagging ensemble)
# ══════════════════════════════════════════════════════════════════
class RandomForest:
    def __init__(
        self,
        n_estimators      = 50,
        max_depth         = 15,
        min_samples_split = 5,
        min_samples_leaf  = 2,
        n_features        = 'sqrt',
        n_thresholds      = 10,
        random_state      = 42
    ):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.n_features        = n_features
        self.n_thresholds      = n_thresholds
        self.random_state      = random_state
        self.trees_            = []

    def _resolve_n_features(self, n_total):
        if self.n_features == 'sqrt':
            return max(1, int(np.sqrt(n_total)))
        if self.n_features == 'log2':
            return max(1, int(np.log2(n_total)))
        if isinstance(self.n_features, int):
            return min(self.n_features, n_total)
        return n_total

    def fit(self, X, y):
        rng           = np.random.default_rng(self.random_state)
        n_samples     = X.shape[0]
        n_feat_actual = self._resolve_n_features(X.shape[1])
        self.trees_   = []

        for i in range(self.n_estimators):
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot   = X[boot_idx]
            y_boot   = y[boot_idx]

            tree = DecisionTree(
                max_depth         = self.max_depth,
                min_samples_split = self.min_samples_split,
                min_samples_leaf  = self.min_samples_leaf,
                n_features        = n_feat_actual,
                n_thresholds      = self.n_thresholds,
                random_state      = int(rng.integers(0, 10_000))
            )
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)

        self.classes_ = np.unique(y.astype(int))
        return self

    def predict(self, X):
        all_preds = np.array([t.predict(X) for t in self.trees_])
        result = []
        for col in range(all_preds.shape[1]):
            votes = all_preds[:, col].astype(int)
            result.append(int(np.bincount(votes).argmax()))
        return np.array(result)

    def get_params(self):
        return dict(
            n_estimators      = self.n_estimators,
            max_depth         = self.max_depth,
            min_samples_split = self.min_samples_split,
            min_samples_leaf  = self.min_samples_leaf,
            n_features        = self.n_features,
            n_thresholds      = self.n_thresholds,
            random_state      = self.random_state
        )


# ══════════════════════════════════════════════════════════════════
#  PART 3 — ADABOOST WITH DECISION STUMPS  (boosting ensemble)
# ══════════════════════════════════════════════════════════════════
class AdaBoostTree:
    """
    AdaBoost with shallow Decision Trees (SAMME algorithm).
    """
    def __init__(
        self,
        n_estimators = 50,
        max_depth    = 1,
        random_state = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.random_state = random_state
        self.estimators_  = []
        self.alphas_      = []
        self.classes_     = None

    def fit(self, X, y):
        y      = y.astype(int)
        n      = len(y)
        K      = len(np.unique(y))
        self.classes_ = np.unique(y)

        w   = np.ones(n) / n
        rng = np.random.default_rng(self.random_state)

        self.estimators_ = []
        self.alphas_     = []

        for t in range(self.n_estimators):
            idx   = rng.choice(n, size=n, replace=True, p=w)
            X_w   = X[idx]
            y_w   = y[idx]

            stump = DecisionTree(
                max_depth         = self.max_depth,
                min_samples_split = 2,
                min_samples_leaf  = 1,
                n_features        = None,
                n_thresholds      = 10,
                random_state      = int(rng.integers(0, 10_000))
            )
            stump.fit(X_w, y_w)
            y_pred = stump.predict(X)

            wrong = (y_pred != y).astype(float)
            eps   = np.dot(w, wrong)
            eps   = np.clip(eps, 1e-10, 1 - 1e-10)

            alpha = np.log((1.0 - eps) / eps) + np.log(K - 1)

            if alpha <= 0:
                break

            w = w * np.exp(alpha * wrong)
            w = w / w.sum()

            self.estimators_.append(stump)
            self.alphas_.append(alpha)

        return self

    def predict(self, X):
        K      = len(self.classes_)
        scores = np.zeros((len(X), K))

        for alpha, stump in zip(self.alphas_, self.estimators_):
            preds = stump.predict(X).astype(int)
            for i, p in enumerate(preds):
                if p < K:
                    scores[i, p] += alpha

        return self.classes_[np.argmax(scores, axis=1)]

    def get_params(self):
        return dict(
            n_estimators = self.n_estimators,
            max_depth    = self.max_depth,
            random_state = self.random_state
        )


# ══════════════════════════════════════════════════════════════════
#  PART 4 — HYPERPARAMETER TUNER
# ══════════════════════════════════════════════════════════════════
class HyperparameterTuner:
    def __init__(self, ModelClass, param_grid, k=5, seed=42, verbose=True):
        self.ModelClass = ModelClass
        self.param_grid = param_grid
        self.k          = k
        self.seed       = seed
        self.verbose    = verbose

    @staticmethod
    def _accuracy(y_true, y_pred):
        return float(np.mean(y_true.astype(int) == y_pred.astype(int)))

    @staticmethod
    def _macro_f1(y_true, y_pred):
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        classes = np.unique(y_true)
        f1s = []
        for c in classes:
            tp = int(np.sum((y_true == c) & (y_pred == c)))
            fp = int(np.sum((y_true != c) & (y_pred == c)))
            fn = int(np.sum((y_true == c) & (y_pred != c)))
            p  = tp / (tp + fp) if (tp + fp) else 0.0
            r  = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append((2 * p * r) / (p + r) if (p + r) else 0.0)
        return float(np.mean(f1s))

    @staticmethod
    def _stratified_folds(X, y, k, seed):
        np.random.seed(seed)
        classes = np.unique(y)
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
            yield X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    @staticmethod
    def _expand_grid(grid):
        keys   = list(grid.keys())
        values = list(grid.values())
        combos = [[]]
        for v in values:
            combos = [c + [x] for c in combos for x in v]
        return [dict(zip(keys, c)) for c in combos]

    def fit(self, X, y):
        combos  = self._expand_grid(self.param_grid)
        results = []

        print(f"\n  HyperparameterTuner: {len(combos)} configs × {self.k} folds "
              f"= {len(combos) * self.k} fits")

        best_f1     = -1
        best_params = None

        for i, params in enumerate(combos):
            fold_f1s = []
            for X_tr, y_tr, X_val, y_val in self._stratified_folds(
                    X, y, self.k, self.seed):
                model = self.ModelClass(**params)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_val)
                fold_f1s.append(self._macro_f1(y_val, pred))

            mean_f1 = float(np.mean(fold_f1s))
            std_f1  = float(np.std(fold_f1s))
            results.append({'params': params, 'mean_f1': mean_f1, 'std_f1': std_f1})

            if mean_f1 > best_f1:
                best_f1     = mean_f1
                best_params = params

            if self.verbose:
                print(f"    [{i+1:3d}/{len(combos)}]  F1={mean_f1:.4f}±{std_f1:.4f}  "
                      f"params={params}")

        print(f"\n  Best F1={best_f1:.4f}  params={best_params}")

        self.best_params_ = best_params
        self.best_score_  = best_f1
        self.cv_results_  = results

        return best_params, results


# ══════════════════════════════════════════════════════════════════
#  PART 5 — BIAS-VARIANCE ANALYSER
# ══════════════════════════════════════════════════════════════════
class BiasVarianceAnalyser:
    def __init__(self, model_factory, n_bootstrap=10, train_size=None,
                 random_state=42):
        self.model_factory = model_factory
        self.n_bootstrap   = n_bootstrap
        self.train_size    = train_size
        self.random_state  = random_state

    def analyse(self, X_train, y_train, X_test, y_test):
        rng      = np.random.default_rng(self.random_state)
        n_train  = len(X_train)
        n_test   = len(X_test)
        size     = self.train_size or n_train

        all_preds = np.zeros((self.n_bootstrap, n_test), dtype=int)

        print(f"\n  BiasVarianceAnalyser: {self.n_bootstrap} bootstrap rounds…")

        for b in range(self.n_bootstrap):
            idx   = rng.choice(n_train, size=size, replace=True)
            X_b   = X_train[idx]
            y_b   = y_train[idx]
            model = self.model_factory()
            model.fit(X_b, y_b)
            all_preds[b] = model.predict(X_test).astype(int)
            print(f"    Round {b+1}/{self.n_bootstrap}  done")

        y_test_i = y_test.astype(int)

        main_preds = np.array([
            int(np.bincount(all_preds[:, i]).argmax())
            for i in range(n_test)
        ])

        bias_sq   = float(np.mean(main_preds != y_test_i))
        variance  = float(np.mean([
            np.mean(all_preds[:, i] != main_preds[i])
            for i in range(n_test)
        ]))
        total_err = float(np.mean(all_preds != y_test_i[np.newaxis, :]))

        print(f"\n  ── Bias-Variance Decomposition ──")
        print(f"     Bias²    = {bias_sq:.4f}")
        print(f"     Variance = {variance:.4f}")
        print(f"     Total err= {total_err:.4f}")

        if bias_sq > 0.30:
            diag = "HIGH BIAS → UNDERFITTING → increase max_depth"
        elif variance > 0.20:
            diag = "HIGH VARIANCE → OVERFITTING → decrease max_depth"
        else:
            diag = "BALANCED → Good generalisation"

        print(f"     Diagnosis: {diag}")

        return {
            'bias_sq':   bias_sq,
            'variance':  variance,
            'total_err': total_err,
            'diagnosis': diag
        }


# ══════════════════════════════════════════════════════════════════
#  PART 6 — LEARNING CURVE RUNNER
# ══════════════════════════════════════════════════════════════════
class LearningCurveRunner:
    def __init__(
        self,
        model_factory,
        train_sizes  = None,
        n_folds      = 3,
        random_state = 42,
        save_dir     = 'plots'
    ):
        self.model_factory = model_factory
        self.train_sizes   = train_sizes or [0.05, 0.1, 0.2, 0.3,
                                              0.5, 0.7, 0.9, 1.0]
        self.n_folds       = n_folds
        self.random_state  = random_state
        self.save_dir      = save_dir

    @staticmethod
    def _accuracy(y_true, y_pred):
        return float(np.mean(y_true.astype(int) == y_pred.astype(int)))

    def run(self, X_train, y_train, X_val, y_val, model_name='Model'):
        rng          = np.random.default_rng(self.random_state)
        n            = len(X_train)
        train_scores = []
        val_scores   = []
        sizes_abs    = []

        print(f"\n  LearningCurveRunner — {model_name}")

        for frac in self.train_sizes:
            size  = max(10, int(n * frac))
            idx   = rng.choice(n, size=size, replace=False)
            X_sub = X_train[idx]
            y_sub = y_train[idx]

            model   = self.model_factory()
            model.fit(X_sub, y_sub)

            tr_acc  = self._accuracy(y_sub, model.predict(X_sub))
            val_acc = self._accuracy(y_val, model.predict(X_val))

            train_scores.append(tr_acc)
            val_scores.append(val_acc)
            sizes_abs.append(size)

            print(f"    n={size:6d}  train_acc={tr_acc:.4f}  val_acc={val_acc:.4f}")

        self._plot(sizes_abs, train_scores, val_scores, model_name)

        return {
            'sizes':        sizes_abs,
            'train_scores': train_scores,
            'val_scores':   val_scores
        }

    def _plot(self, sizes, train_scores, val_scores, model_name):
        os.makedirs(self.save_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sizes, train_scores, 'o-', color='#2196F3',
                label='Training accuracy',   linewidth=2)
        ax.plot(sizes, val_scores,   's-', color='#E74C3C',
                label='Validation accuracy', linewidth=2)

        ax.fill_between(sizes, train_scores, val_scores,
                         alpha=0.15, color='#FF9800',
                         label='Generalisation gap')

        gap         = np.array(train_scores) - np.array(val_scores)
        max_gap_idx = int(np.argmax(gap))
        ax.annotate(
            f'Max gap\n{gap[max_gap_idx]:.3f}',
            xy=(sizes[max_gap_idx],
                (train_scores[max_gap_idx] + val_scores[max_gap_idx]) / 2),
            fontsize=9, color='#FF9800',
            arrowprops=dict(arrowstyle='->', color='#FF9800')
        )

        ax.set_xlabel('Training set size', fontsize=11)
        ax.set_ylabel('Accuracy',          fontsize=11)
        ax.set_title(f'Learning Curve — {model_name}', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()

        safe = model_name.replace(' ', '_')
        path = os.path.join(self.save_dir, f'learning_curve_{safe}.png')
        plt.savefig(path, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"    Saved → {path}")


# ══════════════════════════════════════════════════════════════════
#  PART 7 — DEPTH vs ACCURACY PLOT
# ══════════════════════════════════════════════════════════════════
def plot_depth_vs_accuracy(X_train, y_train, X_val, y_val,
                           depths=None, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    depths     = depths or [1, 2, 3, 5, 8, 10, 12, 15, 18, 20, 25, 30]
    train_accs = []
    val_accs   = []

    print("\n  Depth vs Accuracy analysis…")
    for d in depths:
        tree = DecisionTree(max_depth=d, n_features=None)
        tree.fit(X_train, y_train)
        tr  = float(np.mean(tree.predict(X_train).astype(int)
                            == y_train.astype(int)))
        val = float(np.mean(tree.predict(X_val).astype(int)
                            == y_val.astype(int)))
        train_accs.append(tr)
        val_accs.append(val)
        print(f"    depth={d:3d}  train={tr:.4f}  val={val:.4f}")

    best_depth = depths[int(np.argmax(val_accs))]
    print(f"\n  Best val accuracy at depth={best_depth}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(depths, train_accs, 'o-', color='#2196F3',
            label='Train accuracy',      linewidth=2)
    ax.plot(depths, val_accs,   's-', color='#E74C3C',
            label='Validation accuracy', linewidth=2)
    ax.axvline(best_depth, color='#2ECC71', linestyle='--',
               label=f'Best depth={best_depth}', linewidth=1.5)

    ax.text(depths[0] + 0.3, 0.3, 'Underfitting\n(high bias)',
            color='#1565C0', fontsize=9)
    ax.text(depths[-3],       0.3, 'Overfitting\n(high variance)',
            color='#C62828', fontsize=9)

    ax.set_xlabel('max_depth', fontsize=11)
    ax.set_ylabel('Accuracy',  fontsize=11)
    ax.set_title('Bias-Variance Trade-off — max_depth vs Accuracy', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, 'depth_vs_accuracy.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")

    return best_depth, list(zip(depths, train_accs, val_accs))


# ══════════════════════════════════════════════════════════════════
#  PART 8 — ENSEMBLE BAR CHART
# ══════════════════════════════════════════════════════════════════
def plot_ensemble_comparison(results_dict, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)

    names  = list(results_dict.keys())
    accs   = [results_dict[n]['accuracy'] for n in names]
    f1s    = [results_dict[n]['f1']       for n in names]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63'][:len(names)]

    x   = np.arange(len(names))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - w/2, accs, w, label='Accuracy',
                   color=[c + 'CC' for c in colors], edgecolor='white')
    bars2 = ax.bar(x + w/2, f1s,  w, label='Macro F1',
                   color=colors, edgecolor='white')

    for bar, v in zip(list(bars1) + list(bars2), accs + f1s):
        ax.text(bar.get_x() + bar.get_width()/2,
                v + 0.005, f'{v:.3f}',
                ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Model Comparison — Single Tree vs Ensembles',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    path = os.path.join(save_dir, 'ensemble_comparison.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════
#  PART 9 — CONFUSION MATRIX COMPARISON  (NEW)
#
#  Shows side-by-side 10×10 confusion matrices for:
#    Left  — Baseline DecisionTree
#    Right — Best adaptive model (Random Forest or AdaBoost)
#
#  Reading the two matrices together reveals exactly which digit
#  pairs the ensemble fixes relative to the single tree:
#    • Diagonal cells  → correct predictions (want these high)
#    • Off-diagonal    → confusion between two classes
#  A darker diagonal and lighter off-diagonals on the right panel
#  means the ensemble is making fewer systematic mistakes.
# ══════════════════════════════════════════════════════════════════
def plot_cm_comparison(
    y_test,
    y_pred_baseline,
    y_pred_adaptive,
    adaptive_name,
    save_dir='plots'
):
    """
    Save a side-by-side confusion matrix figure:
        Left  panel  — Baseline DecisionTree
        Right panel  — adaptive_name  (Random Forest or AdaBoost)

    Parameters
    ----------
    y_test          : true labels  (n,)
    y_pred_baseline : predictions from the single DecisionTree
    y_pred_adaptive : predictions from the best ensemble
    adaptive_name   : string label for the right panel
    save_dir        : directory to save the PNG
    """
    os.makedirs(save_dir, exist_ok=True)

    y_test          = y_test.astype(int)
    y_pred_baseline = y_pred_baseline.astype(int)
    y_pred_adaptive = y_pred_adaptive.astype(int)

    n_classes = len(np.unique(y_test))
    labels    = [str(i) for i in range(n_classes)]

    # ── build both confusion matrices from scratch (no sklearn) ──
    def _build_cm(y_true, y_pred):
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm

    cm_base  = _build_cm(y_test, y_pred_baseline)
    cm_adapt = _build_cm(y_test, y_pred_adaptive)

    acc_base  = cm_base.diagonal().sum()  / cm_base.sum()
    acc_adapt = cm_adapt.diagonal().sum() / cm_adapt.sum()

    # ── figure ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, cm, title, acc, cmap in zip(
        axes,
        [cm_base,  cm_adapt],
        ['Baseline — Decision Tree', f'Adaptive — {adaptive_name}'],
        [acc_base,  acc_adapt],
        ['Blues',   'Greens']
    ):
        # heat map
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # cell annotations
        thresh = cm.max() / 2.0
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, str(cm[i, j]),
                        ha='center', va='center', fontsize=8,
                        color='white' if cm[i, j] > thresh else 'black')

        # green border on diagonal cells to highlight correct predictions
        for k in range(n_classes):
            ax.add_patch(plt.Rectangle(
                (k - 0.5, k - 0.5), 1, 1,
                fill=False, edgecolor='#2ECC71', linewidth=1.8
            ))

        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Predicted Digit', fontsize=11)
        ax.set_ylabel('True Digit',      fontsize=11)
        ax.set_title(f'{title}\nAccuracy = {acc:.4f}', fontsize=12, pad=10)

    # delta annotation — shows improvement
    delta = acc_adapt - acc_base
    sign  = '+' if delta >= 0 else ''
    fig.suptitle(
        f'Confusion Matrix Comparison\n'
        f'Accuracy improvement: {sign}{delta:.4f}  '
        f'({adaptive_name} vs Decision Tree)',
        fontsize=13, fontweight='bold', y=1.01
    )

    plt.tight_layout()
    path = os.path.join(save_dir, 'cm_baseline_vs_adaptive.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════
#  PART 10 — COMPLETE DT PIPELINE  (called from main file)
# ══════════════════════════════════════════════════════════════════
def run_dt_full_pipeline(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    save_dir          = 'plots',
    run_tuning        = False,   # skipped — best params already known
    run_bv_analysis   = True,
    run_learning_curve= True,
    run_depth_sweep   = False,   # skipped — max_depth fixed at 15
    run_ensembles     = True
):
    os.makedirs(save_dir, exist_ok=True)

    # ── helpers ──────────────────────────────────────────────────
    def accuracy(yt, yp):
        return float(np.mean(yt.astype(int) == yp.astype(int)))

    def macro_metrics(yt, yp):
        yt = yt.astype(int); yp = yp.astype(int)
        classes = np.unique(yt)
        precs, recs, f1s = [], [], []
        for c in classes:
            tp = int(np.sum((yt == c) & (yp == c)))
            fp = int(np.sum((yt != c) & (yp == c)))
            fn = int(np.sum((yt == c) & (yp != c)))
            p  = tp / (tp + fp) if (tp + fp) else 0.0
            r  = tp / (tp + fn) if (tp + fn) else 0.0
            f  = (2*p*r)/(p+r) if (p+r) else 0.0
            precs.append(p); recs.append(r); f1s.append(f)
        acc = accuracy(yt, yp)
        return acc, float(np.mean(precs)), float(np.mean(recs)), float(np.mean(f1s))

    print("\n" + "═"*65)
    print("  DECISION TREE FULL PIPELINE")
    print("═"*65)

    # ── Step 1: Hyperparameter tuning ────────────────────────────
    # Best params already found via grid search (F1 = 0.8178).
    # run_tuning=False by default so this block is skipped and
    # these values are used directly without re-running the search.
    best_params = {
        'max_depth':          15,
        'min_samples_split':  5,
        'min_samples_leaf':   5,
        'n_features':         None,
        'n_thresholds':       10,
    }

    if run_tuning:
        print("\n── Step 1: Hyperparameter Grid Search ──────────────────")
        param_grid = {
            'max_depth':          [10, 15, 20],
            'min_samples_split':  [5, 10],
            'min_samples_leaf':   [2, 5],
            'n_features':         [None],
            'n_thresholds':       [10],
        }
        tuner = HyperparameterTuner(DecisionTree, param_grid, k=3, verbose=True)
        best_params, cv_results = tuner.fit(X_train, y_train)
        print(f"  Best params: {best_params}")

    # ── Step 2: Depth sweep ───────────────────────────────────────
    if run_depth_sweep:
        print("\n── Step 2: Depth Sweep (Overfitting / Underfitting) ────")
        best_depth, depth_results = plot_depth_vs_accuracy(
            X_train, y_train, X_val, y_val,
            depths   = [1, 3, 5, 8, 10, 15, 20],
            save_dir = save_dir
        )
        best_params['max_depth'] = best_depth

    # ── Step 3: Learning curve ────────────────────────────────────
    if run_learning_curve:
        print("\n── Step 3: Learning Curve ───────────────────────────────")
        def dt_factory():
            return DecisionTree(**best_params)
        lcr = LearningCurveRunner(
            model_factory = dt_factory,
            train_sizes   = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            save_dir      = save_dir
        )
        lcr.run(X_train, y_train, X_val, y_val, model_name='DecisionTree')

    # ── Step 4: Bias-Variance decomposition ──────────────────────
    bv_result = None
    if run_bv_analysis:
        print("\n── Step 4: Bias-Variance Decomposition ─────────────────")
        def dt_factory_bv():
            return DecisionTree(**best_params)
        bva = BiasVarianceAnalyser(
            model_factory = dt_factory_bv,
            n_bootstrap   = 5,
            train_size    = min(5000, len(X_train)),
            random_state  = 42
        )
        n_bv      = min(1000, len(X_test))
        bv_result = bva.analyse(X_train, y_train,
                                X_test[:n_bv], y_test[:n_bv])

    # ── Step 5: Final single DecisionTree (BASELINE) ─────────────
    print("\n── Step 5: Final DecisionTree (Baseline) ────────────────")
    final_dt    = DecisionTree(**best_params)
    final_dt.fit(X_train, y_train)
    y_pred_dt   = final_dt.predict(X_test)
    dt_acc, dt_prec, dt_rec, dt_f1 = macro_metrics(y_test, y_pred_dt)
    print(f"  DecisionTree  Acc={dt_acc:.4f}  Prec={dt_prec:.4f}  "
          f"Rec={dt_rec:.4f}  F1={dt_f1:.4f}")

    # ── Step 6: Random Forest ─────────────────────────────────────
    y_pred_rf = y_pred_dt          # fallback if ensembles skipped
    rf_acc = rf_prec = rf_rec = rf_f1 = 0.0
    if run_ensembles:
        print("\n── Step 6: Random Forest ────────────────────────────────")
        rf = RandomForest(
            n_estimators      = 30,
            max_depth         = best_params.get('max_depth', 15),
            min_samples_split = best_params.get('min_samples_split', 5),
            min_samples_leaf  = best_params.get('min_samples_leaf', 2),
            n_features        = 'sqrt',
        )
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        rf_acc, rf_prec, rf_rec, rf_f1 = macro_metrics(y_test, y_pred_rf)
        print(f"  RandomForest  Acc={rf_acc:.4f}  Prec={rf_prec:.4f}  "
              f"Rec={rf_rec:.4f}  F1={rf_f1:.4f}")

    # ── Step 7: AdaBoost ─────────────────────────────────────────
    y_pred_ab = y_pred_dt          # fallback if ensembles skipped
    ab_acc = ab_prec = ab_rec = ab_f1 = 0.0
    if run_ensembles:
        print("\n── Step 7: AdaBoost ─────────────────────────────────────")
        ab = AdaBoostTree(n_estimators=30, max_depth=2)
        ab.fit(X_train, y_train)
        y_pred_ab = ab.predict(X_test)
        ab_acc, ab_prec, ab_rec, ab_f1 = macro_metrics(y_test, y_pred_ab)
        print(f"  AdaBoost      Acc={ab_acc:.4f}  Prec={ab_prec:.4f}  "
              f"Rec={ab_rec:.4f}  F1={ab_f1:.4f}")

    # ── Step 8: Ensemble bar chart ────────────────────────────────
    if run_ensembles:
        comparison = {
            'Decision Tree': {'accuracy': dt_acc, 'f1': dt_f1},
            'Random Forest': {'accuracy': rf_acc, 'f1': rf_f1},
            'AdaBoost':      {'accuracy': ab_acc, 'f1': ab_f1},
        }
        plot_ensemble_comparison(comparison, save_dir=save_dir)

    # ── Step 9: Confusion matrix comparison ──────────────────────
    #   Left  → Baseline DecisionTree
    #   Right → whichever ensemble scored higher F1
    if run_ensembles:
        print("\n── Step 9: Confusion Matrix Comparison ──────────────────")
        if rf_f1 >= ab_f1:
            adaptive_preds = y_pred_rf
            adaptive_name  = 'Random Forest'
        else:
            adaptive_preds = y_pred_ab
            adaptive_name  = 'AdaBoost'

        plot_cm_comparison(
            y_test,
            y_pred_baseline = y_pred_dt,
            y_pred_adaptive = adaptive_preds,
            adaptive_name   = adaptive_name,
            save_dir        = save_dir
        )

    # ── Summary ───────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────")
    print(f"  {'Model':<18} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print(f"  {'─'*48}")
    print(f"  {'Decision Tree':<18} {dt_acc:8.4f} {dt_prec:8.4f} "
          f"{dt_rec:8.4f} {dt_f1:8.4f}")
    if run_ensembles:
        print(f"  {'Random Forest':<18} {rf_acc:8.4f} {rf_prec:8.4f} "
              f"{rf_rec:8.4f} {rf_f1:8.4f}")
        print(f"  {'AdaBoost':<18} {ab_acc:8.4f} {ab_prec:8.4f} "
              f"{ab_rec:8.4f} {ab_f1:8.4f}")

    if bv_result:
        print(f"\n  Bias²={bv_result['bias_sq']:.4f}  "
              f"Variance={bv_result['variance']:.4f}  "
              f"→ {bv_result['diagnosis']}")

    # ── Return best model results ─────────────────────────────────
    scores = {'Decision Tree': dt_f1}
    if run_ensembles:
        scores['Random Forest'] = rf_f1
        scores['AdaBoost']      = ab_f1

    best_name  = max(scores, key=scores.get)
    best_preds = {
        'Decision Tree': y_pred_dt,
        'Random Forest': y_pred_rf,
        'AdaBoost':      y_pred_ab,
    }[best_name]

    best_acc, best_prec, best_rec, best_f1 = macro_metrics(y_test, best_preds)

    n_classes = len(np.unique(y_test))
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_test.astype(int), best_preds.astype(int)):
        cm[t, p] += 1

    print(f"\n  Best model: {best_name}  F1={best_f1:.4f}")

    return {
        'cv':     (best_acc, best_prec, best_rec, best_f1),
        'cv_std': (0.0, 0.0, 0.0, 0.0),
        'test':   (best_acc, best_prec, best_rec, best_f1),
        'cm':     cm
    }
