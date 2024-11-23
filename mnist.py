import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.src.datasets import mnist
from utils import plot_images_and_classes


def main() -> None:
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    class_names = np.arange(0, 10)
    plot_images_and_classes(train_X, train_y, class_names)


if __name__ == "__main__":
    main()