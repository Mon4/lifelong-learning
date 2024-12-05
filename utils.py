import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def plot_images_and_classes(loader: DataLoader, class_names: list) -> None:
    for batch_id, (data, label) in enumerate(loader):
        fig, ax = plt.subplots(3, 3)
        fig.tight_layout()
        for i in range(9):
            plt.subplot(3, 3, i+1)

            if len(data[i].shape) == 2:
                reshaped_data = data[i].squeeze(1)
                plt.imshow(reshaped_data, cmap=plt.get_cmap('gray'))
            else:
                img_rgb = np.transpose(data[i], (1, 2, 0))
                plt.imshow(img_rgb)

            if isinstance(label[i], list):
                train_y_i = label[i][0]
            else:
                train_y_i = label[i]

            plt.title(class_names[train_y_i])
            plt.axis('off')
        plt.show()
        break


def load_datasets(dataset_name: str = 'mnist', batch_size: int = 64) -> (DataLoader, DataLoader, DataLoader):
    # Define the common transform
    transform = transforms.Compose([transforms.ToTensor()])

    # Set the dataset path and load the dataset
    if dataset_name == 'mnist':
        path = '../mnist_data'
        train_dataset = datasets.MNIST(path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(path, train=False, download=True, transform=transform)
    elif dataset_name == 'cifar10':
        path = '../cifar10_data'
        train_dataset = datasets.CIFAR10(path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(path, train=False, download=True, transform=transform)
    else:
        raise ValueError(f'Dataset {dataset_name} not recognized')

    # Create train DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Split test set into validation and test sets
    val_size = int(0.5 * len(test_dataset))
    test_size = len(test_dataset) - val_size
    val_data, test_data = random_split(test_dataset, [val_size, test_size])

    # Create validation and test DataLoaders
    validation_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, validation_loader