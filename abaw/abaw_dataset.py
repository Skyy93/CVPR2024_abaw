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
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os
import soundfile as sf

cv2.setNumThreads(2)


class HumeDatasetTrain(Dataset):

    def __init__(self, data_folder, label_file=None, model=None):
        super().__init__()
        self.data_folder = data_folder
        self.label_file = pd.read_csv(label_file)
        self.vision_model = model[0]
        self.audio_model = model[1]

        if self.vision_model != 'linear':
            self.data_config = timm.data.resolve_model_data_config(self.vision_model)
            self.transform = A.Compose([
                A.Resize(height=self.data_config['input_size'][1], width=self.data_config['input_size'][2]),
                A.Normalize(mean=self.data_config['mean'], std=self.data_config['std']),
                ToTensorV2(),
            ])

        if self.audio_model != 'linear':
            self.processor = AutoProcessor.from_pretrained(self.audio_model)

    def __getitem__(self, index):
        row = self.label_file.iloc[index]

        if self.vision_model == 'linear':
            vit_file_path = f"{self.data_folder}vit/{str(int(row['Filename'])).zfill(5)}.pkl"
            with open(vit_file_path, 'rb') as file:
                vision = torch.mean(torch.tensor(pickle.load(file)), dim=0)
        else:
            vision = self.process_images(index)
        
        if self.audio_model == 'linear':
            wav2vec2_file_path = f"{self.data_folder}wav2vec2/{str(int(row['Filename'])).zfill(5)}.pkl"
            with open(wav2vec2_file_path, 'rb') as file:
                audio = torch.mean(torch.tensor(pickle.load(file)), dim=0)
        else:
            audio = self.process_audio(row['Filename'])
        labels = torch.tensor(
            row[['Admiration', 'Amusement', 'Determination', 'Empathic Pain', 'Excitement', 'Joy']].values,
            dtype=torch.float)
        return audio, vision, labels

    def process_images(self, index):
        try:
            img_folder_path = f"{self.data_folder}face_images/{str(int(index)).zfill(5)}/"
            img_files = sorted(os.listdir(img_folder_path))
            selected_indices = np.linspace(0, len(img_files) - 1, min(50, len(img_files)), dtype=int)
            images = []
            for idx in selected_indices:
                img_path = os.path.join(img_folder_path, img_files[idx])
                img = Image.open(img_path).convert('RGB')
                images.append(self.transform(image=np.array(img))['image'])
            # Add black images if there are less than 50 images
            while len(images) < 50:
                black_img = Image.new('RGB', (160, 160))
                images.append(self.transform(image=np.array(black_img))['image'])

            return torch.stack(images)
        except Exception as e:
            images = []
            print(e)
            while len(images) < 1: # correct when face images are there
                black_img = Image.new('RGB', (160, 160))
                images.append(self.transform(image=np.array(black_img))['image'])
            print(f"No image found for index: {index}")
            return torch.stack(images)

    def process_audio(self, filename):
        audio_file_path = f"{self.data_folder}audio/{str(int(filename)).zfill(5)}.mp3"
        try:
            audio_data, sr = sf.read(audio_file_path)
        except Exception as e:
            print(f"Error processing audio file {audio_file_path}: {e}")
            audio_data = np.zeros(128, dtype=np.float32)

        return audio_data


    def __len__(self):
        return len(self.label_file)

    def collate_fn(self, batch):
        audio_data, vision_data, labels_data = zip(*batch)
        audio_data_padded = self.processor(audio_data, padding=True, sampling_rate=16000, return_tensors="pt", max_length=1024, truncation=True)
        vision_stacked = torch.stack(vision_data)
    
        labels_stacked = torch.stack(labels_data)
    
        return audio_data_padded, vision_stacked, labels_stacked


class HumeDatasetEval(Dataset):

    def __init__(self, data_folder, label_file=None, model=None):
        super().__init__()
        self.data_folder = data_folder
        self.label_file = pd.read_csv(label_file)
        self.vision_model = model[0]
        self.audio_model = model[1]

        if self.vision_model != 'linear':
            self.data_config = timm.data.resolve_model_data_config(self.vision_model)
            self.transform = A.Compose([
                A.Resize(height=self.data_config['input_size'][1], width=self.data_config['input_size'][2]),
                A.Normalize(mean=self.data_config['mean'], std=self.data_config['std']),
                ToTensorV2(),
            ])
        if self.audio_model != 'linear':
            self.processor = AutoProcessor.from_pretrained(self.audio_model)

    def __getitem__(self, index):
        row = self.label_file.iloc[index]

        if self.vision_model == 'linear':
            vit_file_path = f"{self.data_folder}vit/{str(int(row['Filename'])).zfill(5)}.pkl"
            with open(vit_file_path, 'rb') as file:
                vision = torch.mean(torch.tensor(pickle.load(file)), dim=0)
        else:
            vision = self.process_images(index)

        if self.audio_model == 'linear':
            wav2vec2_file_path = f"{self.data_folder}wav2vec2/{str(int(row['Filename'])).zfill(5)}.pkl"
            with open(wav2vec2_file_path, 'rb') as file:
                audio = torch.mean(torch.tensor(pickle.load(file)), dim=0)
        else:
            audio = self.process_audio(row['Filename'])
        labels = torch.tensor(
            row[['Admiration', 'Amusement', 'Determination', 'Empathic Pain', 'Excitement', 'Joy']].values,
            dtype=torch.float)
        return audio, vision, labels

    def process_images(self, index):
        try:
            img_folder_path = f"{self.data_folder}face_images/{str(int(index)).zfill(5)}/"
            img_files = sorted(os.listdir(img_folder_path))
            selected_indices = np.linspace(0, len(img_files) - 1, min(50, len(img_files)), dtype=int)
            images = []
            for idx in selected_indices:
                img_path = os.path.join(img_folder_path, img_files[idx])
                img = Image.open(img_path).convert('RGB')
                images.append(self.transform(image=np.array(img))['image'])

            # Add black images if there are less than 50 images
            while len(images) < 50:
                black_img = Image.new('RGB', (160, 160))
                images.append(self.transform(image=np.array(black_img))['image'])

            return torch.stack(images)
        except:
            images = []
            while len(images) < 1: # TODO correct when faceimage are there
                black_img = Image.new('RGB', (160, 160))
                images.append(self.transform(image=np.array(black_img))['image'])
            #print(f"No image found for index: {index}")
            return torch.stack(images)

    def process_audio(self, filename):
        audio_file_path = f"{self.data_folder}audio/{str(int(filename)).zfill(5)}.mp3"
        try:
            audio_data, sr = sf.read(audio_file_path)
        except Exception as e:
            print(f"Error processing audio file {audio_file_path}: {e}")
            audio_data = np.zeros((128,), dtype=np.float32)

        return audio_data

    def __len__(self):
        return len(self.label_file)

    def collate_fn(self, batch):
        audio_data, vision_data, labels_data = zip(*batch)
        audio_data_padded = self.processor(audio_data, padding=True, sampling_rate=16000, return_tensors="pt", max_length=1024, truncation=True)
        vision_stacked = torch.stack(vision_data)
    
        labels_stacked = torch.stack(labels_data)
    
        return audio_data_padded, vision_stacked, labels_stacked
