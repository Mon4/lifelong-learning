import pickle
import torch
from matplotlib import pyplot as plt

from siamese.model import SiameseNetwork

model = SiameseNetwork()
model.load_state_dict(torch.load('model.pth', weights_only=True))
model.eval()

with open('losses.pkl', 'rb') as f:
    losses = pickle.load(f)

plt.plot(range(0, len(losses)), losses, label='train')
plt.title('Loss function')
plt.show()

print(losses)
print(model)