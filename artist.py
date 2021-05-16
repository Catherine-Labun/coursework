from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import numpy as np

class Artist(Dataset):
    def __init__(self, transforms=None):
        self.data = None
        self.transforms = transforms

        images = pd.read_csv('images.txt', sep=' ',names=['img_id', 'image_path'])
        class_label = pd.read_csv('images_class_labels.txt', sep=' ', names=['img_id', 'label'])
        artists = pd.read_csv('artists.txt', sep=',', names=['label', 'author'])

        self.data = images.merge(class_label, on='img_id')
        self.data = self.data.merge(artists, on='label')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = sample.image_path
        label = sample.label
        image = Image.open(path)
        image = np.array(image)

        if len(image.shape) == 2:
            image = np.stack((image, )*3, axis=-1)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label