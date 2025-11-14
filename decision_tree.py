import numpy
from torchvision.datasets import CIFAR10
import confusion_matrix as confusion
import load_dataset as ld

TRAINING_SET_SIZE = 500
TEST_SET_SIZE = 100

# ------------------- Node class definition ------------------- #
class Node:
    def __init__(self, gini, samples, value, feature_index=None, threshold=None, left=None, right=None):
        self.gini = gini
        self.samples = samples
        self.value = value
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right


# ------------------- Model class definition ------------------- #
class DecisionTreeClassifierScratch:
    def __init__(self, max_depth=50, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_classes = None

    def gini(self, y):
        classes, counts = numpy.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - numpy.sum(probs ** 2)

    def best_split(self, X, y):
        m, n = X.shape
        if m < self.min_samples_split:
            return None, None

        best_gini = 1e9
        best_feat = None
        best_thresh = None

        for feat_idx in range(n):
            thresholds = numpy.unique(X[:, feat_idx])

            for t in thresholds:
                left_mask = X[:, feat_idx] <= t
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                left_gini = self.gini(y[left_mask])
                right_gini = self.gini(y[right_mask])
                weighted = (left_mask.sum()/m)*left_gini + (right_mask.sum()/m)*right_gini

                if weighted < best_gini:
                    best_gini = weighted
                    best_feat = feat_idx
                    best_thresh = t

        return best_feat, best_thresh

    def build(self, X, y, depth=0):
        num_samples = len(y)
        value = numpy.bincount(y, minlength=self.n_classes)
        gini_imp = self.gini(y)

        node = Node(gini_imp, num_samples, value)

        if depth < self.max_depth:
            feat_idx, threshold = self.best_split(X, y)
            if feat_idx is not None:
                left_mask = X[:, feat_idx] <= threshold
                right_mask = ~left_mask

                node.feature_index = feat_idx
                node.threshold = threshold
                node.left = self.build(X[left_mask], y[left_mask], depth + 1)
                node.right = self.build(X[right_mask], y[right_mask], depth + 1)

        return node

    def fit(self, X, y):
        self.n_classes = len(numpy.unique(y))
        self.root = self.build(X, y)

    def predict_one(self, x, node):
        if node.feature_index is None:
            return numpy.argmax(node.value)

        if x[node.feature_index] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    def predict(self, X):
        return numpy.array([self.predict_one(x, self.root) for x in X])


# ------------------- Run the model ------------------- #
if __name__ == "__main__":
    # Load CIFAR-10 subset
    print("Loading dataset...")
    x_train, y_train, x_test, y_test = ld.load_cifar_dataset()

    depths = [1, 2, 5, 10, 20, 30, 50] # Sizes greater than 50 take a long time and offer only small changes to accuracy

    # Run the model for different max depths
    for d in depths:
        print(f"Training decision tree (depth {d})...")
        tree = DecisionTreeClassifierScratch(max_depth=d)
        tree.fit(x_train, y_train)

        # Predict
        print("Evaluating...")
        y_pred = tree.predict(x_test)

        acc = numpy.mean(y_pred == y_test)
        print(f"Decision tree accuracy on 100 images: {acc:.4f}")

        # Confusion matrix
        cm = confusion.confusion_matrix(y_test, y_pred, num_classes=len(numpy.unique(y_test)))
        print("Confusion Matrix:")
        print(cm)

        # Print per-class accuracy
        print("Per class accuracy:")
        class_acc = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(class_acc):
            print(f"Class {i}: {acc:.2f}")
