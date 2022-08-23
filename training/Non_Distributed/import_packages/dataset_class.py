"""
Dataset class is overrriding the torch dataset class.

Author : Rishav

"""
import numpy as np
import os
import random
import glob
from PIL import Image
import torchvision.transforms.functional as TF
import torch
# https://stackoverflow.com/a/20749411/6642287


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class for loading the data in dataloader.

    Overriden the Pytorch Dataset class to build a custom data loader to load csv files # noqa
    Parameters
    -----------
    list_IDs : Accepts the dataset distribution(train/val/test)
    labels : Accepts the corresponding labels

    """

    def __init__(self, list_IDs, labels, root_dir, train_transform=False, valid_transform=False, test_transform=True): # noqa
        """To instantiate labels and the data distribution."""
        self.labels = labels
        self.list_IDs = list_IDs
        self.dir = root_dir
        self.race_folder = glob.glob(os.path.join(root_dir, '*.JPG'))
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform

    def __len__(self):
        """Return a Sample from the dataset given an index."""
        # return len(self.list_IDs)
        return 50

    def train_transforms(self, image):
        """To train transformation."""
        if random.random() > 0.5:
            image = TF.hflip(image)
            image = TF.vflip(image)
        # image = TF.adjust_brightness(image, 2)
        # image = TF.adjust_contrast(image, 2)
        # image = TF.adjust_saturation(image, 2)
        # image = TF.adjust_hue(image, 0)
        image = TF.to_tensor(image)
        image = TF.normalize(image, [-0.4389, -0.5410, -0.5801], [1.0257, 0.9810, 0.9896]) # noqa
        return image

    def valid_transforms(self, image):
        """To valid transformation."""
        if random.random() > 0.5:
            image = TF.hflip(image)
            image = TF.vflip(image)
        # image = TF.adjust_brightness(image, 2)
        # image = TF.adjust_contrast(image, 2)
        # image = TF.adjust_saturation(image, 2)
        # image = TF.adjust_hue(image, 0)
        image = TF.to_tensor(image)
        image = TF.normalize(image, [-0.4389, -0.5410, -0.5801],[1.0257, 0.9810, 0.9896]) # noqa
        return image

    def test_transforms(self, image):
        """ To test tranformation """
        image = TF.to_tensor(image)
        image = TF.normalize(image, [-0.4389, -0.5410, -0.5801], [1.0257, 0.9810, 0.9896]) # noqa
        return image

    def __getitem__(self, index):
        """Generate one sample of data."""
        ID = self.list_IDs[index]
        image = Image.open(os.path.join(self.dir, ID))
        y = self.labels[ID]
        if self.train_transform:
            image = self.train_transforms(image)
        elif self.valid_transform:
            image = self.valid_transforms(image)
        else:
            image = self.test_transforms(image)
        img = np.array(image)
        return img, y


if __name__ == 'main':
    partition, labels = train_val_to_ids(csv_file='Combined_Files_without_label5.csv', stratify_columns='labels', random_state=42) # noqa
    training_set = Dataset(partition['train_set'],labels,root_dir = '/Users/swastik/ophthalmology/Project_Quality_Assurance/Mod_AEON_data') # noqa
    train_loader = torch.utils.data.DataLoader(training_set, shuffle=True)
