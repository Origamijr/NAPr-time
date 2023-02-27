import os
import torch
import numpy as np
import pandas as pd
from torch.utils import data
from tqdm import tqdm
from torch.utils.data import Dataset, random_split
from torchvision.transforms import Normalize, Compose

from config import data_config, training_config
from preprocessing import get_directory

DATA_PARAMS = data_config()
TRAIN_PARAMS = training_config()

class H5SpecSeqDataset(Dataset):
    """
    Dataset for sequences of 2D features from an HDF5 file
    """

    def __init__(self, 
            hdf_file=DATA_PARAMS['preprocessed_data'], 
            bulk=DATA_PARAMS['single_file'], 
            transform=Compose([torch.tensor, Normalize(-57.6, 19)])):
        label_key = DATA_PARAMS['hdf_label_key']

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


class H5WaveSeqDataset(Dataset):
    """
    Dataset for sequences of waveforms from an HDF5 file
    """

    def __init__(self, 
            hdf_file=DATA_PARAMS['preprocessed_data'], 
            bulk=DATA_PARAMS['single_file'], 
            transform=Compose([torch.tensor]), 
            concat=2): # number of consecutive frames to concatenate
        label_key = DATA_PARAMS['hdf_label_key']

        self.labels = pd.read_hdf(hdf_file + '.h5' if bulk else os.path.join(hdf_file, label_key), key=label_key)
        
        dfs = []
        for label in tqdm(self.labels, desc='Reading Dataset', smoothing=0.1):
            dfs += [pd.read_hdf(hdf_file + '.h5' if bulk else os.path.join(hdf_file, label_key), key=label)]
        self.df = pd.concat(dfs, ignore_index=True)
        self.concat = concat
        self.transform = transform

    
    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        wave_seq = row['wave']
        wave = np.array([wave_seq[i:i+self.concat,:].flatten() for i in range(wave_seq.shape[0] - self.concat + 1)])
        label = get_directory(row['file'])

        if self.transform is not None:
            spec = self.transform(wave)

        return spec, self.get_label_index(label)


    def get_label_index(self, label):
        return self.labels[self.labels == label].index[0]

    
    def get_label(self, idx):
        return self.labels[idx]


def split_dataset(dataset, split=TRAIN_PARAMS['train_val_test_split'], split_seed=TRAIN_PARAMS['split_seed']):
    lengths = [int(len(dataset) * p) for p in split]
    lengths[0] += len(dataset) - sum(lengths)
    return random_split(dataset, lengths, torch.Generator().manual_seed(split_seed))