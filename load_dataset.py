import numpy
from torchvision.datasets import CIFAR10

TRAINING_SET_SIZE = 500
TEST_SET_SIZE = 100

# ------------------- Load data from the CIFAR10 dataset ------------------- #
def load_cifar_dataset():
    """
    Loads the CIFAR 10 dataset and prepares 500 training images 
    and 100 test images for each of the 10 image classes.
    """
    train_ds = CIFAR10(root='./data', train=True, download=True)
    test_ds  = CIFAR10(root='./data', train=False, download=True)

    x_train = numpy.stack([numpy.array(img) for img, _ in train_ds])
    y_train = numpy.array([label for _, label in train_ds])
    x_test  = numpy.stack([numpy.array(img) for img, _ in test_ds])
    y_test  = numpy.array([label for _, label in test_ds])

    y_train = y_train.flatten();
    y_test = y_test.flatten();

    x_train_select = []
    y_train_select = []
    x_test_select = []
    y_test_select = []

    for c in range(10):  # classes 0–9
        class_mask = numpy.where(y_train == c)[0]
        chosen = class_mask[:TRAINING_SET_SIZE]
        x_train_select.append(x_train[chosen])
        y_train_select.append(y_train[chosen])

    for c in range(10):  # classes 0–9
        class_mask = numpy.where(y_train == c)[0]
        chosen = class_mask[:TEST_SET_SIZE] 
        x_test_select.append(x_test[chosen])
        y_test_select.append(y_test[chosen])
    
    x_train_select = numpy.vstack(x_train_select)
    y_train_select = numpy.hstack(y_train_select)
    x_test_select = numpy.vstack(x_test_select)
    y_test_select = numpy.hstack(y_test_select)

    print(f"Training samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")

    x_train_select = x_train_select.reshape((x_train_select.shape[0], -1))
    x_test_select = x_test_select.reshape((x_test_select.shape[0], -1))

    return x_train_select, y_train_select, x_test_select, y_test_select