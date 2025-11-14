import numpy
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from torchvision.datasets import CIFAR10
import load_dataset as ld
import confusion_matrix as confusion

x_train, y_train, x_test, y_test = ld.load_cifar_dataset()

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
print(f"\nTest accuracy: {accuracy:.4f}")

cm = confusion.confusion_matrix(y_test, y_pred, num_classes=len(numpy.unique(y_test)))
print("Confusion matrix:")
print(cm)

# Print per-class accuracy
print("Per class accuracy:")
class_acc = cm.diagonal() / cm.sum(axis=1)
for i, acc in enumerate(class_acc):
    print(f"Class {i}: {acc:.2f}")