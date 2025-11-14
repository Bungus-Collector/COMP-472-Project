import numpy
from torchvision.datasets import CIFAR10
import load_dataset as ld
import confusion_matrix as confusion

TRAINING_SET_SIZE = 500
TEST_SET_SIZE = 100

# ------------------- Model class definition ------------------- #
class GaussianNaiveBayes:
    def __init__(self, epsilon=1e-9):
        self.epsilon = epsilon

    def fit(self, X, y):
        self.classes = numpy.unique(y)
        self.means = {}
        self.vars = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = numpy.mean(X_c, axis=0)
            self.vars[c] = numpy.var(X_c, axis=0) + self.epsilon
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def _pdf_log(self, class_idx, X):
        mean = self.means[class_idx]
        var = self.vars[class_idx]
        log_prob = -0.5 * numpy.sum(numpy.log(2.0 * numpy.pi * var))
        log_prob -= 0.5 * numpy.sum(((X - mean) ** 2) / var, axis=1)
        return log_prob

    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_probs = numpy.zeros((n_classes, n_samples))

        for idx, c in enumerate(self.classes):
            prior = numpy.log(self.priors[c])
            class_conditional = self._pdf_log(c, X)
            log_probs[idx, :] = prior + class_conditional

        class_indices = numpy.argmax(log_probs, axis=0)
        return self.classes[class_indices]


# ------------------- Run the model ------------------- #
if __name__ == "__main__":
    x_train, y_train, x_test, y_test = ld.load_cifar_dataset()

    # Flatten and normalize
    x_train = x_train.reshape((x_train.shape[0], -1)) / 255.0
    x_test  = x_test.reshape((x_test.shape[0], -1)) / 255.0

    print("Flattened shapes:", x_train.shape, x_test.shape)

    # Train & predict
    gnb = GaussianNaiveBayes()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)

    # Evaluate
    accuracy = numpy.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.4f}")

    # Confusion matrix
    cm = confusion.confusion_matrix(y_test, y_pred, num_classes=len(numpy.unique(y_test)))
    print("Confusion Matrix:")
    print(cm)

    # Print per-class accuracy
    print("Per class accuracy:")
    class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_acc):
        print(f"Class {i}: {acc:.2f}")
