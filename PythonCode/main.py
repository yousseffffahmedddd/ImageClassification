from knn import KNN
import Knn_Preprocessing
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ── Load Dataset ──────────────────────────────────────────────────────────────

base_path = "C:\\ImageClassification\\dataset"

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)

x_train = load_mnist_images(base_path + '/train-images-idx3-ubyte/train-images-idx3-ubyte')
y_train = load_mnist_labels(base_path + '/train-labels-idx1-ubyte/train-labels-idx1-ubyte')
x_test  = load_mnist_images(base_path + '/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
y_test  = load_mnist_labels(base_path + '/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

# ── Binary Classification ─────────────────────────────────────────────────────
y_train = (y_train == 0).astype(int)
y_test  = (y_test  == 0).astype(int)

# ── Preprocessing ─────────────────────────────────────────────────────────────

x_train_flat = Knn_Preprocessing.flatten_images(x_train)
x_test_flat  = Knn_Preprocessing.flatten_images(x_test)

x_train_norm = Knn_Preprocessing.normalize_images(x_train_flat)
x_test_norm  = Knn_Preprocessing.normalize_images(x_test_flat)

# ── Configuration ─────────────────────────────────────────────────────────────

# FEATURE FUNCTION
FEATURE_FN = Knn_Preprocessing.pca_fn(n_components=30)

# BALANCE FUNCTION 
BALANCE_FN    = None
BALANCE_LABEL = "None"   # label printed in fold output, change to match

# K-FOLD settings
K_FOLDS = 5    # number of folds, typical: 3, 5, 10
KNN_K   = 3    # number of neighbours, try odd numbers: 1, 3, 5, 7


if FEATURE_FN is not None:
    x_train_feat, x_test_feat = FEATURE_FN(x_train_norm, x_test_norm)
else:
    x_train_feat, x_test_feat = x_train_norm, x_test_norm

# Apply balancing to final training set
if BALANCE_FN is not None:
    x_train_balanced, y_train_balanced = BALANCE_FN(x_train_feat, y_train)
else:
    x_train_balanced, y_train_balanced = x_train_feat, y_train

# ── K-Fold Cross Validation ───────────────────────────────────────────────────

print("\n── K-Fold Cross Validation ──")

accuracies = Knn_Preprocessing.k_fold_evaluate(
    x_train_norm,    
    y_train,
    k=K_FOLDS,
    knn_k=KNN_K,
    feature_fn=FEATURE_FN,
    balance_fn=BALANCE_FN,
    balance_label=BALANCE_LABEL
)

print("\nK-Fold Results:")
print(f"  Average Accuracy : {np.mean(accuracies)*100:.2f}%")
print(f"  Std Dev          : {np.std(accuracies)*100:.2f}%")

# ── Final Model ───────────────────────────────────────────────────────────────

print("\n── Final Model Training ──")

knn_model = KNN(k=KNN_K)
knn_model.fit(x_train_balanced, y_train_balanced)
y_pred = knn_model.predict(x_test_feat)

# ── Evaluation ────────────────────────────────────────────────────────────────

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not 0", "0"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()