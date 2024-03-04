import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import torch
from tqdm import tqdm
import time
import pickle

cv2.setNumThreads(2)


class HumeDatasetTrain(Dataset):

    def __init__(self, data_folder, label_file=None):
        super().__init__()
        self.data_folder = data_folder
        self.label_file = pd.read_csv(label_file)

    def __getitem__(self, index):
        row = self.label_file.iloc[index]

        wav2vec2_file_path = f"{self.data_folder}/wav2vec2/{row['filename'].zfill(5)}.pkl"
        with open(wav2vec2_file_path, 'rb') as file:
            wav2vec2 = pickle.load(file)

        vit_file_path = f"{self.data_folder}/vit/{row['filename'].zfill(5)}.pkl"
        with open(vit_file_path, 'rb') as file:
            vit = pickle.load(file)

        # Extract the required labels and convert them to a tensor
        labels = torch.tensor(
            row[['Admiration', 'Amusement', 'Determination', 'Empathic Pain', 'Excitement', 'Joy']].values,
            dtype=torch.float)

        return wav2vec2, vit, labels

    def __len__(self):
        return len(self.label_file)
        

            
       
class HumeDatasetEval(Dataset):

    def __init__(self, data_folder, label_file=None):
        super().__init__()
        self.data_folder = data_folder
        self.label_file = pd.read_csv(label_file)

    def __getitem__(self, index):
        row = self.label_file.iloc[index]

        wav2vec2_file_path = f"{self.data_folder}/wav2vec2/{row['filename'].zfill(5)}.pkl"
        with open(wav2vec2_file_path, 'rb') as file:
            wav2vec2 = pickle.load(file)

        vit_file_path = f"{self.data_folder}/vit/{row['filename'].zfill(5)}.pkl"
        with open(vit_file_path, 'rb') as file:
            vit = pickle.load(file)

        # Extract the required labels and convert them to a tensor
        labels = torch.tensor(
            row[['Admiration', 'Amusement', 'Determination', 'Empathic Pain', 'Excitement', 'Joy']].values,
            dtype=torch.float)

        return wav2vec2, vit, labels

    def __len__(self):
        return len(self.label_file)




