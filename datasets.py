from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import os


class AudioEncoderDataset(Dataset):
    def __init__(self, chunks_folder, features_folder, format='npy'):
        self.audio_path = chunks_folder
        self.features_path = features_folder
        self.file_names = [filename.split('.')[0] for filename in os.listdir(chunks_folder)]
        self.format = format

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        if self.format == 'npy':
            audio = np.load(self.audio_path + file_name + '.npy')
        elif self.format == 'npz':
            audio = np.load(self.audio_path + file_name + '.npz')['audio']
        else:
            raise Exception("Unknown format of the input data")
        audio_tensor = torch.tensor(audio.reshape(1, -1)).float()

        labels = np.load(self.features_path + file_name + '.npz')
        cqccs = torch.tensor(labels['cqcc']).float()
        lfccs = torch.tensor(labels['lfcc']).float()
        mfccs = torch.tensor(labels['mfcc']).float()
        imfccs = torch.tensor(labels['imfcc']).float()
        return audio_tensor, cqccs, lfccs, mfccs, imfccs


class AudioClassifierDataset(Dataset):
    def __init__(self, chunks_folder, labels_file, format='npy'):
        self.format = format

        self.audio_path = chunks_folder
        audio_labels_csv = pd.read_csv(labels_file, sep=' ', header=None)
        file_names = [filename.split('.')[0] for filename in os.listdir(chunks_folder)]

        self.audio_labels = pd.DataFrame(file_names, columns=['file'])
        self.audio_labels['rawname'] = ["_".join(file_name.split('_')[:-1]) for file_name in self.audio_labels.file]
        self.audio_labels['rawlabel'] = self.audio_labels.merge(audio_labels_csv, how='left', left_on='rawname', right_on=1)[4]
        self.audio_labels['label'] = [0 if label == 'bonafide' else 1 for label in self.audio_labels['rawlabel']]

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        file_name = self.audio_labels.loc[idx, 'file']
        label = self.audio_labels.loc[idx, 'label']

        if self.format == 'npy':
            audio = np.load(self.audio_path + file_name + '.npy')
        elif self.format == 'npz':
            audio = np.load(self.audio_path + file_name + '.npz')['audio']
        else:
            raise Exception("Unknown format of the input data")
        x = torch.tensor(audio.reshape(1, -1)).float()
        y = torch.tensor(label)

        return x, y
