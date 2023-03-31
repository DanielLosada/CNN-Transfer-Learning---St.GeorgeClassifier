import os

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):

    def __init__(self, images_path, labels_path, transform=None):
        super().__init__()
        self.images_path = images_path
        self.df_labels = pd.read_csv(labels_path)
        self.transform = transform 
        
    def __len__(self):
        return len(self.df_labels)

    def __getitem__(self, idx):
        suite_id, sample_id, code, value, character = self.df_labels.loc[idx]
        path = os.path.join(self.images_path, f"input_{suite_id}_{sample_id}_{code}.jpg")
        sample = Image.open(path)
        if self.transform:
            sample = self.transform(sample)
        return sample, code-1