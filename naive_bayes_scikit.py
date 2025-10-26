from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from torchvision.datasets import CIFAR10
import numpy as np

TRAINING_SET_SIZE = 500
TEST_SET_SIZE = 100

# Step 1: Load the CIFAR-10 dataset
# will download into ./data if not present
train_ds = CIFAR10(root='./data', train=True, download=True)
test_ds  = CIFAR10(root='./data', train=False, download=True)

# convert to numpy arrays (shape: N, 32, 32, 3)
x_train = np.stack([np.array(img) for img, _ in train_ds])
y_train = np.array([label for _, label in train_ds])

x_test  = np.stack([np.array(img) for img, _ in test_ds])
y_test  = np.array([label for _, label in test_ds])

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Step 2: Use only a subset of the data
x_train = x_train[:TRAINING_SET_SIZE]
y_train = y_train[:TRAINING_SET_SIZE]
x_test = x_test[:TEST_SET_SIZE]
y_test = y_test[:TEST_SET_SIZE]

print(f"Training samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")

# Step 3: Flatten images (32x32x3 = 3072 features per image)
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

# Step 4: Normalize pixel values (optional, helps PCA and GNB)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Step 5: Apply PCA for dimensionality reduction
pca_components = 50  # can tweak this (20–100 works well)
pca = PCA(n_components=pca_components, whiten=True, random_state=42)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

print(f"Reduced feature size: {x_train_pca.shape[1]}")

# Step 6: Train Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(x_train_pca, y_train)

# Step 7: Predict and evaluate
y_pred = gnb.predict(x_test_pca)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))