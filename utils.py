import numpy as np
from matplotlib import pyplot as plt


def plot_images_and_classes(train_X: np.ndarray, train_y: np.ndarray, class_names: list) -> None:
    fig, ax = plt.subplots(3, 3)
    fig.tight_layout()
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(train_X[i])
        if isinstance(train_y[i], list):
            train_y_i = train_y[i][0]
        else:
            train_y_i = train_y[i]
        plt.title(class_names[train_y_i])
        plt.axis('off')
    plt.show()