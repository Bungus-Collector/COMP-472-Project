import numpy
from sklearn.tree import DecisionTreeClassifier
import confusion_matrix as confusion
import load_dataset as ld

depths = [1, 2, 5, 10, 20, 30, 50]

x_train, y_train, x_test, y_test = ld.load_cifar_dataset()

for d in depths:
    maxDepth = d
    print(f"Training decision tree (depth {d})...")

    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=50
    )

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    acc = numpy.mean(y_pred == y_test)
    print("Scikit decision tree accuracy:", acc)

    cm = confusion.confusion_matrix(y_test, y_pred, num_classes=len(numpy.unique(y_test)))
    print("Confusion Matrix:")
    print(cm)

    print("Per class accuracy:")
    class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_acc):
        print(f"Class {i}: {acc:.2f}")