import numpy as np
import matplotlib
matplotlib.use('Agg')          # Bug 6 fix — must come BEFORE pyplot import
import matplotlib.pyplot as plt
from logisticregression import LogisticRegression


# ══════════════════════════════════════════════════════════════
#  LOAD MNIST
# ══════════════════════════════════════════════════════════════
base_path = "C:/Users/Lenovo/Desktop/mnist/project/dataset/"

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)

x_train = load_mnist_images(base_path + 'train-images-idx3-ubyte/train-images-idx3-ubyte')
y_train = load_mnist_labels(base_path + 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
x_test  = load_mnist_images(base_path + 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
y_test  = load_mnist_labels(base_path + 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

print("Raw shapes:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# ══════════════════════════════════════════════════════════════
#  STEP 1 — NORMALISE  (divide by 255, flatten)
# ══════════════════════════════════════════════════════════════
X_train_norm = x_train.reshape(-1, 784) / 255.0   # float32 in [0, 1]
X_test_norm  = x_test.reshape(-1, 784)  / 255.0


# ══════════════════════════════════════════════════════════════
#  STEP 2 — ONE-VS-ALL LABEL ENCODING
#  digit 0  →  y = 1  (positive class)
#  digits 1–9 →  y = 0  (negative class)
# ══════════════════════════════════════════════════════════════
y_train_bin = (y_train == 0).astype(int)   # 1 where digit is 0, else 0
y_test_bin  = (y_test  == 0).astype(int)

pos_tr = y_train_bin.sum()
neg_tr = (y_train_bin == 0).sum()
pos_te = y_test_bin.sum()
neg_te = (y_test_bin  == 0).sum()

print(f"\nTrain — Positive (digit 0): {pos_tr}  |  Negative (others): {neg_tr}")
print(f"Test  — Positive (digit 0): {pos_te}  |  Negative (others): {neg_te}")
print(f"Imbalance ratio (train): 1 : {neg_tr // pos_tr}")


# ══════════════════════════════════════════════════════════════
#  STANDARDSCALER  
# ══════════════════════════════════════════════════════════════
class StandardScaler:
    """
    Computes mean and std from training data only.
    Applies the same transform to val/test — no data leakage.
    """
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)          # shape (784,)
        self.std_  = np.std(X,  axis=0) + 1e-8   # +1e-8 avoids divide-by-zero
                                                   # on constant (background) pixels
    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ══════════════════════════════════════════════════════════════
#  STEP 3 — DOWNSAMPLE FUNCTION
# ══════════════════════════════════════════════════════════════
def downsample_to_balance(X, y):
    """
    Reduces the majority class to match the minority class size.

    Bug 2 was here: the original code hardcoded
        yn = np.zeros(len(Xn))
        y_bal = np.hstack((np.ones(len(X0)), yn))
    which threw away the actual y values entirely.

    Fix: separate X and y together, index them together,
    then stack the real y slices — never manufacture labels.
    """
    X_pos = X[y == 1];  y_pos = y[y == 1]   # digit 0 samples  → label 1
    X_neg = X[y == 0];  y_neg = y[y == 0]   # all other digits → label 0

    n_pos = len(y_pos)                        # minority size (~5923 in train)
    np.random.seed(42)
    idx = np.random.choice(len(X_neg), size=n_pos, replace=False)

    X_neg_down = X_neg[idx]
    y_neg_down = y_neg[idx]                  # ← real labels carried through

    X_bal = np.vstack([X_pos, X_neg_down])
    y_bal = np.hstack([y_pos, y_neg_down])   # ← stacked from real arrays

    perm = np.random.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


# ══════════════════════════════════════════════════════════════
#  METRICS FUNCTION
# ══════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred):
 
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))

    acc  = float(np.mean(y_true == y_pred))

    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return acc, prec, rec, f1, tp, fn, tn, fp


# ══════════════════════════════════════════════════════════════
#  PIPELINE FUNCTION

# ══════════════════════════════════════════════════════════════
def train_and_evaluate(X_tr, y_tr, X_te, y_te, desc=""):
    """
    Bug 3 was: scaler fitted once on all 60k samples at the top,
    before balancing. This means the scaler statistics came from
    a different distribution than what the model trains on.

    Fix: fit the scaler here, after receiving the final X_tr
    (which is already filtered/balanced by the caller).
    The scaler sees exactly the same data the model trains on.
    """
    print(f"\n{'='*52}")
    print(f"  {desc}")
    print(f"{'='*52}")
    print(f"  Train — pos: {y_tr.sum()}  neg: {(y_tr==0).sum()}")
    print(f"  Test  — pos: {y_te.sum()}  neg: {(y_te==0).sum()}")


    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)    # fit on train
    X_te_sc = scaler.transform(X_te)        # same stats applied to test

    # Train
    model = LogisticRegression(lr=0.1, epochs=500)
    model.fit(X_tr_sc, y_tr)

    # Predict
    y_pred = model.predict(X_te_sc)

    acc, prec, rec, f1, tp, fn, tn, fp = compute_metrics(y_te, y_pred)

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}   (of all predicted-positive, how many truly are?)")
    print(f"  Recall    : {rec:.4f}   (of all true digit-0, how many did we find?)")
    print(f"  F1-Score  : {f1:.4f}   (harmonic mean of precision and recall)")
    print(f"\n  Confusion Matrix:")
    print(f"    TP={tp:5d}  FN={fn:5d}   (actual digit 0)")
    print(f"    FP={fp:5d}  TN={tn:5d}   (actual other digits)")


    plt.figure(figsize=(7, 4))
    plt.plot(model.losses, color='steelblue', linewidth=1.5)
    plt.title(f"Loss Curve — {desc}")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    safe_name = desc.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(f"loss_{safe_name}.png", dpi=120)
    plt.close()   # always close — prevents memory leaks across multiple calls
    print(f"\n  Loss curve saved → loss_{safe_name}.png")

    return acc, prec, rec, f1


# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 1 — IMBALANCED
#  Full training set: ~5923 positives vs ~54077 negatives (1:9 ratio)
#  Scaler fitted on the full imbalanced training data
# ══════════════════════════════════════════════════════════════
acc_u, prec_u, rec_u, f1_u = train_and_evaluate(
    X_train_norm, y_train_bin,
    X_test_norm,  y_test_bin,
    desc="Imbalanced (One-vs-All, full training set)"
)


# ══════════════════════════════════════════════════════════════
#  EXPERIMENT 2 — BALANCED (downsampled)
#  Downsample negatives to match ~5923 positives → 1:1 ratio
#  Scaler fitted AFTER balancing 
# ══════════════════════════════════════════════════════════════
X_tr_bal, y_tr_bal = downsample_to_balance(X_train_norm, y_train_bin)

print(f"\nBalanced train set: {X_tr_bal.shape}")
print(f"  pos: {y_tr_bal.sum()}  neg: {(y_tr_bal==0).sum()}")

acc_b, prec_b, rec_b, f1_b = train_and_evaluate(
    X_tr_bal,    y_tr_bal,
    X_test_norm, y_test_bin,   # same test set — fair comparison
    desc="Balanced (downsampled, One-vs-All)"
)


# ══════════════════════════════════════════════════════════════
#  FINAL COMPARISON
# ══════════════════════════════════════════════════════════════
print("\n" + "="*52)
print("  FINAL COMPARISON")
print("="*52)
print(f"  {'Metric':<12} {'Imbalanced':>12} {'Balanced':>12}  {'Winner'}")
print(f"  {'-'*50}")
print(f"  {'Accuracy':<12} {acc_u:12.4f} {acc_b:12.4f}  {'<-- misleading on imbalanced' if acc_u > acc_b else ''}")
print(f"  {'Precision':<12} {prec_u:12.4f} {prec_b:12.4f}")
print(f"  {'Recall':<12} {rec_u:12.4f} {rec_b:12.4f}  {'<-- key metric here'}")
print(f"  {'F1-Score':<12} {f1_u:12.4f} {f1_b:12.4f}  {'<-- overall winner'}")
# ================================
# 🔹 LOSS CURVE
# ================================


plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
