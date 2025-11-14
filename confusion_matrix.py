import numpy

# ------------------- Confusion matrix ------------------- #
def confusion_matrix(y_true, y_pred, num_classes):
    """
    Compute confusion matrix from true and predicted labels.
    Returns a (num_classes x num_classes) matrix where
    rows = true labels, columns = predicted labels.
    """
    # Basically counts how often class i (rows) is predicted as class j (columns)
    cm = numpy.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm