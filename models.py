import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = x.view(-1, 64 * 7 * 7)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


def get_fisher_diag(model, dataset):
    fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    model.eval()

    for inputs, labels in dataset:
        model.zero_grad()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            fisher[name] += param.grad.data ** 2 / len(dataset)

    return {name: f.detach() for name, f in fisher.items()}

def get_ewc_loss(model, fisher, old_params):
    ewc_loss = 0
    for name, param in model.named_parameters():
        ewc_loss += (fisher[name] * (param - old_params[name]) ** 2).sum()
    return ewc_loss


def train_model(model, train_loader:DataLoader, epoch:int=10, old_params=None, fisher=None):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for epoch in range(epoch):
        print(f'Epoch {epoch+1}')
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            if old_params is not None and fisher is not None:
                ewc_loss = get_ewc_loss(model, fisher, old_params)
                loss += 0.01 * ewc_loss  # lambda for ewc

            loss.backward()
            optimizer.step()