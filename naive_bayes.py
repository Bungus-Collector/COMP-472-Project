import numpy as np
from torchvision.datasets import CIFAR10

TRAINING_SET_SIZE = 500
TEST_SET_SIZE = 100

# ------------------- Model class definition ------------------- #
class GaussianNaiveBayes:
    def __init__(self, epsilon=1e-9):
        self.epsilon = epsilon

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.vars = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0) + self.epsilon
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def _pdf_log(self, class_idx, X):
        mean = self.means[class_idx]
        var = self.vars[class_idx]
        log_prob = -0.5 * np.sum(np.log(2.0 * np.pi * var))
        log_prob -= 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
        return log_prob

    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_probs = np.zeros((n_classes, n_samples))

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[c])
            class_conditional = self._pdf_log(c, X)
            log_probs[idx, :] = prior + class_conditional

        class_indices = np.argmax(log_probs, axis=0)
        return self.classes[class_indices]

# ------------------- Confusion matrix ------------------- #
def confusion_matrix(y_true, y_pred, num_classes):
    """
    Compute confusion matrix from true and predicted labels.
    Returns a (num_classes x num_classes) matrix where
    rows = true labels, columns = predicted labels.
    """
    # Basically counts how often class i (rows) is predicted as class j (columns)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# ------------------- Run the model ------------------- #
if __name__ == "__main__":
    train_ds = CIFAR10(root='./data', train=True, download=True)
    test_ds  = CIFAR10(root='./data', train=False, download=True)

    x_train = np.stack([np.array(img) for img, _ in train_ds])
    y_train = np.array([label for _, label in train_ds])
    x_test  = np.stack([np.array(img) for img, _ in test_ds])
    y_test  = np.array([label for _, label in test_ds])

    print("Original shapes:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # Take subset
    x_train = x_train[:TRAINING_SET_SIZE]
    y_train = y_train[:TRAINING_SET_SIZE]
    x_test  = x_test[:TEST_SET_SIZE]
    y_test  = y_test[:TEST_SET_SIZE]

    # Flatten and normalize
    x_train = x_train.reshape((x_train.shape[0], -1)) / 255.0
    x_test  = x_test.reshape((x_test.shape[0], -1)) / 255.0

    print("Flattened shapes:", x_train.shape, x_test.shape)

    # Train & predict
    gnb = GaussianNaiveBayes()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)

    # Evaluate
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, num_classes=len(np.unique(y_test)))
    print("Confusion Matrix:")
    print(cm)

    # Print per-class accuracy
    print("Per class accuracy:")
    class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_acc):
        print(f"Class {i}: {acc:.2f}")
