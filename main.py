import torch
from models import SimpleCNN, get_fisher_diag, train_model
from utils import plot_images_and_classes, load_datasets


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load
    mnist_train_loader, mnist_test_loader, mnist_validation_loader = load_datasets('mnist', 10)
    cifar_train_loader, cifar_test_loader, cifar_validation_loader = load_datasets('cifar10', 10)
    mnist_class_names = range(10)
    cifar_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # plot
    # plot_images_and_classes(mnist_train_loader, mnist_class_names)
    # plot_images_and_classes(cifar_train_loader, cifar_class_names)

    model = SimpleCNN()
    old_params = {name: param.data.clone() for name, param in model.named_parameters()}
    fisher_matrix = get_fisher_diag(model, mnist_train_loader)

    train_model(model=model, loader=mnist_train_loader, epoch=2, old_params=old_params, fisher=fisher_matrix)

    print('hhh')

if __name__ == "__main__":
    main()