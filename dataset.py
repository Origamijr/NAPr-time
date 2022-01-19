import os
import torch
import pandas as pd
from torch.utils import data
from tqdm import tqdm
from torch.utils.data import Dataset, random_split

from config import CONFIG
from preprocessing import get_directory

class H5SpecSeqDataset(Dataset):
    """
    Dataset for sequences of 2D features from an HDF5 file
    """

    def __init__(self, hdf_file=CONFIG['preprocessing']['destination'], bulk=True, transform=None):
        label_key = CONFIG['preprocessing']['hdf_label_key']

        self.labels = pd.read_hdf(hdf_file + '.h5' if bulk else os.path.join(hdf_file, label_key), key=label_key)
        
        dfs = []
        for label in tqdm(self.labels, desc='Reading Dataset', smoothing=0.1):
            dfs += [pd.read_hdf(hdf_file + '.h5' if bulk else os.path.join(hdf_file, label_key), key=label)]
        self.df = pd.concat(dfs, ignore_index=True)
        self.transform = transform

    
    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        spec = row['magnitude']
        label = get_directory(row['file'])

        if self.transform is not None:
            spec = self.transform(spec)

        return spec, self.get_label_index(label)


    def get_label_index(self, label):
        return self.labels[self.labels == label].index[0]

    
    def get_label(self, idx):
        return self.labels[idx]


def split_dataset(dataset, split=CONFIG['training']['train_val_test_split'], split_seed=CONFIG['training']['split_seed']):
    lengths = [int(len(dataset) * p) for p in split]
    lengths[0] += len(dataset) - sum(lengths)
    return random_split(dataset, lengths, torch.Generator().manual_seed(split_seed))