from knn import KNN
from linearsvm import SVM
import os
import numpy as np
import matplotlib.pyplot as plt
#load mnist dataset
base_path = "C:/Users/Lenovo/Desktop/mnist/project/dataset/"

def load_mnist_images(filename):
    with open(filename, 'rb') as f:   # ✅ NOT gzip
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:   # ✅ NOT gzip
        return np.frombuffer(f.read(), np.uint8, offset=8)
    
x_train = load_mnist_images(base_path + 'train-images-idx3-ubyte/train-images-idx3-ubyte')
y_train = load_mnist_labels(base_path + 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
x_test = load_mnist_images(base_path + 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
y_test = load_mnist_labels(base_path + 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

print("Loaded:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print("Unique labels:", np.unique(y_train))
print("Sample image shape:", x_train[0])

for i in range(20):
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.show()