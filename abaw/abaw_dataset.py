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
import timm
from transformers import AutoProcessor

cv2.setNumThreads(2)


class HumeDatasetTrain(Dataset):

    def __init__(self, data_folder, label_file=None, model=None):
        super().__init__()
        self.data_folder = data_folder
        self.label_file = pd.read_csv(label_file)
        self.vision_model = model[0]
        self.audio_model = model[0]

        if self.vision_model != 'linear':
            data_config = timm.data.resolve_model_data_config(self.vision_model)

        if self.audio_model != 'linear':
            self.processor = AutoProcessor(self.audio_model)



    def __getitem__(self, index):
        row = self.label_file.iloc[index]

        wav2vec2_file_path = f"{self.data_folder}wav2vec2/{str(int(row['Filename'])).zfill(5)}.pkl"
        with open(wav2vec2_file_path, 'rb') as file:
            wav2vec2 = torch.mean(torch.tensor(pickle.load(file)), dim=0)

        vit_file_path = f"{self.data_folder}vit/{str(int(row['Filename'])).zfill(5)}.pkl"
        with open(vit_file_path, 'rb') as file:
            vit = torch.mean(torch.tensor(pickle.load(file)), dim=0)

        # Extract the required labels and convert them to a tensor
        labels = torch.tensor(
            row[['Admiration', 'Amusement', 'Determination', 'Empathic Pain', 'Excitement', 'Joy']].values,
            dtype=torch.float)

        return wav2vec2, vit, labels

    def __len__(self):
        return len(self.label_file)

#     def get_config(self, ):
#         data_config = timm.data.resolve_model_data_config(self.model)
#         return data_config
# TODO: needed for later
            
       
class HumeDatasetEval(Dataset):

    def __init__(self, data_folder, label_file=None, model=None):
        super().__init__()
        self.data_folder = data_folder
        self.label_file = pd.read_csv(label_file)

    def __getitem__(self, index):
        row = self.label_file.iloc[index]

        wav2vec2_file_path = f"{self.data_folder}wav2vec2/{str(int(row['Filename'])).zfill(5)}.pkl"
        with open(wav2vec2_file_path, 'rb') as file:
            wav2vec2 = torch.mean(torch.tensor(pickle.load(file)), dim=0)

        vit_file_path = f"{self.data_folder}vit/{str(int(row['Filename'])).zfill(5)}.pkl"
        with open(vit_file_path, 'rb') as file:
            vit = torch.mean(torch.tensor(pickle.load(file)), dim=0)

        # Extract the required labels and convert them to a tensor
        labels = torch.tensor(
            row[['Admiration', 'Amusement', 'Determination', 'Empathic Pain', 'Excitement', 'Joy']].values,
            dtype=torch.float)

        return wav2vec2, vit, labels

    def __len__(self):
        return len(self.label_file)




