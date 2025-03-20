# Load the the dataset from raw image folders
import os
import datetime

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from torch import optim, device, cuda, amp
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from siamese.load_data import CustomDataset
from siamese.model import SiameseNetwork, ContrastiveLoss
import torch.nn.functional as F
import pickle

# train the model
def train(epochs, max_lr, model, train_dl, opt_func=optim.SGD):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    cuda.empty_cache()
    losses = []
    epoch_losses = []

    optimizer = opt_func(model.parameters(), max_lr)
    # one cycle learning rate scheduler
    t_start = datetime.datetime.now()
    print('time start: ', t_start)

    for epoch in range(1, epochs+1):
        print(f'epoch {epoch}')
        for batch_idx, data in enumerate(train_dl):
            print(f'\tbatch {batch_idx}')
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

            optimizer.zero_grad()
            # forward
            with amp.autocast('cuda'):
                output1, output2 = model(img0, img1)
                loss = criterion(output1, output2, label)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            losses.append(loss.item())
            cuda.empty_cache()

            # adam step
            optimizer.step()
        print(f'Cost at epoch {epoch} is {sum(losses) / len(losses)}')
        epoch_losses.append(sum(losses) / len(losses))
        print(datetime.datetime.now() - t_start)
    return model, epoch_losses


# def imshow(img):
#     img = img.permute(1, 2, 0)  # Change (C, H, W) -> (H, W, C)
#     plt.imshow(img.numpy())
#     plt.axis('off')
#     plt.show()

def test(model, test_dl):
    model.eval()

    with (torch.no_grad()):
        # count = 0
        for i, data in enumerate(test_dl):
            cuda.empty_cache()
            x0, x1, labels = data
            output1, output2 = model(x0.to(device), x1.to(device))

            eucledian_distance = F.pairwise_distance(output1, output2, eps=1e-6)

        eucledian_distance = [round(dist.item(), 4) for dist in eucledian_distance]
        text_label = ['not close locations' if lbl == 0 else 'close locations' for lbl in labels]

        fig, axes = plt.subplots(2, 3, figsize=(9, 6))

        for j in range(x0.shape[0]):
            img1 = np.transpose(x0[j], (1, 2, 0))
            img2 = np.transpose(x1[j], (1, 2, 0))
            axes[0, j].imshow(img1)
            axes[0, j].axis("off")
            axes[1, j].set_title(f"label: {labels[j]}\n"
                                 f"{text_label[j]}\n"
                                 f"euclidean distance: {eucledian_distance[j]}")
            axes[1, j].imshow(img2)
            axes[1, j].axis("off")
        plt.show()

        # count = count + 1
        # if count == 10:
        #     break

# TRAIN
# train_dir = "../data/train/DataSet_GOPRO_RGB_train"
#
# device = device('cuda' if cuda.is_available else 'cpu')
# model = SiameseNetwork()
# model = model.to(device)
#
# criterion = ContrastiveLoss()
#
# train_ds = CustomDataset(train_dir)

#
# train_dl = DataLoader(train_ds, shuffle=True, num_workers=0, pin_memory=True, batch_size=8)
#
# epochs = 3
# max_lr = 1e-4
# opt_func = optim.Adam
#
# model, losses = train(epochs, max_lr, model, train_dl, opt_func)
#
# with open('losses.pkl', 'wb') as f:
#     pickle.dump(losses, f)
#
# torch.save(model.state_dict(), 'model.pth')

# TEST
device = device('cuda' if cuda.is_available else 'cpu')

model = SiameseNetwork().to(device)
model.load_state_dict(torch.load('model.pth', weights_only=True))
model.eval()

test_dir = "../data/test/DataSet_GOPRO_RGB_test1"
test_ds = CustomDataset(test_dir)
test_dl = DataLoader(test_ds, shuffle=False, num_workers=0, pin_memory=True, batch_size=8)

torch.cuda.empty_cache()
test(model, test_dl)
