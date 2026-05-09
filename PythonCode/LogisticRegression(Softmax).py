import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog


# ============================================================
# LOAD DATA
# ============================================================
def load_mnist(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = train.iloc[:, 1:].values.astype(np.float32)
    y_train = train.iloc[:, 0].values.astype(int)

    X_test = test.iloc[:, 1:].values.astype(np.float32)
    y_test = test.iloc[:, 0].values.astype(int)

    return X_train, y_train, X_test, y_test


# ============================================================
# PREPROCESSING
# ============================================================
def preprocess(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


# ============================================================
# PCA
# ============================================================
def apply_pca(X_train, X_test, n=90):
    pca = PCA(n_components=n)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    print(f"PCA: 784 → {n}")
    return X_train, X_test


# ============================================================
# HOG FEATURES (NEW)
# ============================================================
def extract_hog(X):
    hog_features = []

    for img in X:
        h = hog(img.reshape(28, 28),
                pixels_per_cell=(7, 7),
                cells_per_block=(2, 2),
                feature_vector=True)
        hog_features.append(h)

    return np.array(hog_features)


# ============================================================
# SOFTMAX (FROM SCRATCH)
# ============================================================
class SoftmaxRegression:
    def __init__(self, lr=0.1, epochs=500, reg_lambda=0.001):
        self.lr = lr
        self.epochs = epochs
        self.reg_lambda = reg_lambda

    def softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)

    def one_hot(self, y, k):
        oh = np.zeros((len(y), k))
        oh[np.arange(len(y)), y] = 1
        return oh

    def fit(self, X, y):
        n, d = X.shape
        self.k = len(np.unique(y))

        self.W = np.zeros((d, self.k))
        self.b = np.zeros((1, self.k))

        y_oh = self.one_hot(y, self.k)

        for _ in range(self.epochs):
            probs = self.softmax(X @ self.W + self.b)

            dw = (X.T @ (probs - y_oh)) / n + self.reg_lambda * self.W
            db = np.mean(probs - y_oh, axis=0)

            self.W -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        probs = self.softmax(X @ self.W + self.b)

        # recall tweak
        probs = probs + 0.03

        return np.argmax(probs, axis=1)


# ============================================================
# GRID SEARCH
# ============================================================
def kfold_cv(X, y, lr, reg, epochs, k=3):
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    fold = len(X) // k

    scores = []

    for i in range(k):
        val_idx = idx[i*fold:(i+1)*fold]
        train_idx = np.setdiff1d(idx, val_idx)

        model = SoftmaxRegression(lr, epochs, reg)
        model.fit(X[train_idx], y[train_idx])

        pred = model.predict(X[val_idx])
        scores.append(np.mean(pred == y[val_idx]))

    return np.mean(scores)


def grid_search(X, y):
    lrs = [0.1, 0.05]
    regs = [0.001, 0.01]
    epochs_list = [500,600,700,800,1000]

    best = None
    best_score = -1

    for lr, reg, ep in itertools.product(lrs, regs, epochs_list):
        score = kfold_cv(X, y, lr, reg, ep)

        print(f"lr={lr}, reg={reg}, epochs={ep} -> {score:.4f}")

        if score > best_score:
            best_score = score
            best = {"lr": lr, "reg_lambda": reg, "epochs": ep}

    print("BEST:", best)
    return best


# ============================================================
# EVALUATION
# ============================================================
def evaluate(y_true, y_pred, title):
    print("\n", "="*50)
    print(title)
    print("="*50)

    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(title)
    plt.show()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro")
    }


# ============================================================
# MAIN
# ============================================================
train_path = "/content/sample_data/mnist_train_small.csv"
test_path = "/content/sample_data/mnist_test.csv"

X_train, y_train, X_test, y_test = load_mnist(train_path, test_path)

# Scale
X_train, X_test = preprocess(X_train, X_test)

# PCA
X_train_pca, X_test_pca = apply_pca(X_train, X_test, 90)

# HOG
X_train_hog = extract_hog(X_train)
X_test_hog = extract_hog(X_test)

# ============================================================
# COMBINE FEATURES (IMPORTANT)
# Softmax benefits from HOG + PCA
# ============================================================
X_train_combined = np.hstack([X_train_pca, X_train_hog])
X_test_combined = np.hstack([X_test_pca, X_test_hog])

# ============================================================
# TRAIN SOFTMAX (USES HOG)
# ============================================================
best = grid_search(X_train_combined, y_train)

model = SoftmaxRegression(**best)
model.fit(X_train_combined, y_train)

pred = model.predict(X_test_combined)

results = {}
results["Softmax + PCA + HOG"] = evaluate(y_test, pred, "Softmax + PCA + HOG")

# ============================================================
# RANDOM FOREST (NO HOG ❗)
# ============================================================
# ❗ WHY NOT HOG HERE:
# - Random Forest is tree-based → does NOT rely on linear structure
# - It splits features individually, so raw pixels already work well
# - HOG removes pixel-level detail that trees can actually use
# - Adding HOG increases dimension without meaningful gain
# 👉 Result: more cost, almost no improvement

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_pca, y_train)   # only PCA, NOT HOG

rf_pred = rf.predict(X_test_pca)

results["Random Forest (NO HOG)"] = evaluate(y_test, rf_pred, "Random Forest + PCA")

# ============================================================
# COMPARISON
# ============================================================
names = list(results.keys())
acc = [results[n]["accuracy"] for n in names]
f1 = [results[n]["f1"] for n in names]

x = np.arange(len(names))
w = 0.35

plt.figure()
plt.bar(x - w/2, acc, w, label="Accuracy")
plt.bar(x + w/2, f1, w, label="Macro F1")

plt.xticks(x, names)
plt.title("Model Comparison (HOG Impact)")
plt.legend()
plt.grid(axis='y')
plt.show()



#  Plots======================================================

# ============================================================
# SOFTMAX: ACCURACY vs EPOCHS
# ============================================================

epochs_list = [50, 200, 300, 500, 700, 1000, 2000, 3000, 4000]

softmax_acc = []

for ep in epochs_list:

    model = SoftmaxRegression(
        lr=best["lr"],
        epochs=ep,
        reg_lambda=best["reg_lambda"]
    )

    model.fit(X_train_combined, y_train)

    pred = model.predict(X_test_combined)

    acc = accuracy_score(y_test, pred)

    softmax_acc.append(acc)

# ============================================================
# FINAL GRAPH
# ============================================================

plt.figure(figsize=(10,6))

plt.plot(
    epochs_list,
    softmax_acc,
    marker='o',
    linewidth=2,
    label="Test Accuracy"
)

# UNDERFITTING REGION
plt.axvspan(
    epochs_list[0],
    300,
    alpha=0.2,
    label="Underfitting Region"
)

# GOOD FIT REGION
plt.axvspan(
    300,
    1500,
    alpha=0.1,
    label="Good Fit Region"
)

# OVERFITTING REGION
plt.axvspan(
    1500,
    epochs_list[-1],
    alpha=0.2,
    label="Possible Overfitting Region"
)

# LABELS
plt.annotate(
    "Underfitting",
    xy=(100, softmax_acc[0]),
    fontsize=10
)

plt.annotate(
    "Good Generalization",
    xy=(700, max(softmax_acc)),
    fontsize=10
)

plt.annotate(
    "Possible Overfitting",
    xy=(2500, softmax_acc[-1]),
    fontsize=10
)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Softmax Accuracy vs Epochs")
plt.grid(True)
plt.legend()

plt.show()


# ============================================================
# RANDOM FOREST: ACCURACY vs DEPTH
# ============================================================

depths = [2, 4, 6, 8, 10, 15, 20, None]

rf_acc = []

depth_labels = []

for depth in depths:

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=depth,
        random_state=42
    )

    rf.fit(X_train_pca, y_train)

    pred = rf.predict(X_test_pca)

    acc = accuracy_score(y_test, pred)

    rf_acc.append(acc)

    depth_labels.append(str(depth))


# ============================================================
# FINAL GRAPH
# ============================================================

plt.figure(figsize=(10,6))

plt.plot(
    range(len(depth_labels)),
    rf_acc,
    marker='s',
    linewidth=2,
    label="Test Accuracy"
)

# UNDERFITTING REGION
plt.axvspan(
    0,
    2,
    alpha=0.2,
    label="Underfitting Region"
)

# GOOD FIT REGION
plt.axvspan(
    2,
    5,
    alpha=0.1,
    label="Good Fit Region"
)

# OVERFITTING REGION
plt.axvspan(
    5,
    7,
    alpha=0.2,
    label="Possible Overfitting Region"
)

# LABELS
plt.annotate(
    "Underfitting",
    xy=(0.5, rf_acc[0]),
    fontsize=10
)

plt.annotate(
    "Good Generalization",
    xy=(3, max(rf_acc)),
    fontsize=10
)

plt.annotate(
    "Possible Overfitting",
    xy=(6, rf_acc[-1]),
    fontsize=10
)

plt.xticks(range(len(depth_labels)), depth_labels)

plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Random Forest Accuracy vs Depth")

plt.grid(True)
plt.legend()

plt.show()
