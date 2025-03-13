import os
import random
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

#preprocessing and loading the data set
class CustomDataset(Dataset):
    def __init__(self,training_dir=None, transform=None):
        self.transform = transform
        self.train_dir = training_dir
        self.image_paths = os.listdir(self.train_dir)
        self.labels = [path.split('_')[1] for path in self.image_paths]
        self.data = list(zip(self.image_paths, self.labels))


    def __getitem__(self, index):
        img1_path, label1 = self.data[index]
        img1_full_path = os.path.join(self.train_dir, img1_path)

        # Choose a second image: same class (positive) or different class (negative)
        if random.random() > 0.5:  # 50% chance of same class
            same_class_indices = [i for i, lbl in enumerate(self.labels) if lbl == label1 and i != index]
            img2_idx = random.choice(same_class_indices) if same_class_indices else index
        else:  # Different class
            different_class_indices = [i for i, lbl in enumerate(self.labels) if lbl != label1]
            img2_idx = random.choice(different_class_indices) if different_class_indices else index

        img2_path, label2 = self.data[img2_idx]
        img2_full_path = os.path.join(self.train_dir, img2_path)

        img1 = read_image(str(img1_full_path)).float() / 255.
        img2 = read_image(str(img2_full_path)).float() / 255.

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # Label: 1 if same class, 0 if different class
        target = torch.tensor(int(label1 == label2), dtype=torch.float32)

        return img1, img2, target  # Return two images and similarity label

    def __len__(self):
        return len(self.image_paths)


# path = "../data/train/DataSet_GOPRO_RGB_train"
# siamese = CustomDataset(training_dir=path)
# print(set(siamese.labels))
# img, label = siamese.__getitem__(1)
# print(label)
# path1 = Path(os.path.basename(train_labels[0]))
# print(path1.stem)

