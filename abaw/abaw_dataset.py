import cv2
import imageio_ffmpeg
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
import abaw.utils
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence
from pathlib import Path

cv2.setNumThreads(2)


class HumeDatasetTrain(Dataset, abaw.utils.AverageMeter):

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
            self.processor_vision = AutoProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

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
        return audio, vision, labels, self.avg

    def process_images(self, index):
        try:
            img_folder_path = f"{self.data_folder}face_images/{str(int(index)).zfill(5)}/"
            img_files = sorted(os.listdir(img_folder_path), key=lambda x: x.zfill(15))
            meta = next(imageio_ffmpeg.read_frames(f"{self.data_folder}raw/{str(int(index)).zfill(5)}.mp4"))
            fps_est = len(img_files)/meta['duration']
            if 'Thumbs.db' in img_files:
                img_files.remove('Thumbs.db')
            selected_indices = np.linspace(0, len(img_files) - 1, min(12*5, max(1, round(5/fps_est*len(img_files)))), dtype=int)
            images = []
            for idx in selected_indices:#range(len(img_files[:12*5])):
                img_path = os.path.join(img_folder_path, img_files[idx])
                img = np.array(Image.open(img_path))#.convert('RGB')#.resize((160, 160))
                #images.append(self.transform(image=np.array(img))['image'])
                images.append(torch.tensor(img))
            #self.update(1-len(images)/50)
            # Add black images if there are less than 50 images
            while len(images) < 1:
                black_img = Image.new('RGB', (224, 224))
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
            if sr != 16000:
                print(audio_file_path)
                raise ValueError
        except Exception as e:
            print(f"Error processing audio file {audio_file_path}: {e}")
            audio_data = np.zeros(128, dtype=np.float32)
        self.update(1 - len(audio_data[:12*sr])/len(audio_data))
        return audio_data[:12*sr]


    def __len__(self):
        return len(self.label_file)
        #return int(2*len(self.label_file)/3)

    def collate_fn(self, batch):
        audio_data, vision_data, labels_data, avg = zip(*batch)
        audio_data_padded = self.processor(audio_data, padding=True, sampling_rate=16000, return_tensors="pt", truncation=True, max_length=12*16000, return_attention_mask=True)
        lengths, permutation = audio_data_padded['attention_mask'].sum(axis=1).sort(descending=True)
        audio_packed = pack_padded_sequence(audio_data_padded['input_values'][permutation], lengths.cpu().numpy(), batch_first=True)  # 'input_features' for w2v2-bert
        # assumption: audio lengths match vision lengths; it does not hold.
        vision_data = [self.processor_vision(x, return_tensors='pt')['pixel_values'] for x in vision_data]
        vision_packed = pack_sequence([vision_data[x] for x in permutation], enforce_sorted=False)
    
        labels_stacked = torch.stack([labels_data[x] for x in permutation])
    
        return audio_packed, vision_packed, labels_stacked, np.mean(avg)


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
            self.processor_vision = AutoProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    def __getitem__(self, index):
        #row = self.label_file.iloc[int(2*len(self.label_file)/3)+index]
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
            img_files = sorted(os.listdir(img_folder_path), key=lambda x: x.zfill(15))
            meta = next(imageio_ffmpeg.read_frames(f"{self.data_folder}raw/{str(int(index)).zfill(5)}.mp4"))
            fps_est = len(img_files)/meta['duration']
            if 'Thumbs.db' in img_files:
                img_files.remove('Thumbs.db')
            selected_indices = np.linspace(0, len(img_files) - 1, min(12*5, max(1, round(5/fps_est*len(img_files)))), dtype=int)
            images = []
            for idx in selected_indices:  # range(len(img_files[:12*5])):
                img_path = os.path.join(img_folder_path, img_files[idx])
                img = np.array(Image.open(img_path))#.convert('RGB')#.resize((160, 160))
                #images.append(self.transform(image=np.array(img))['image'])
                images.append(torch.tensor(img))

            # Add black images if there are less than 50 images
            while len(images) < 1:
                black_img = Image.new('RGB', (160, 160))
                images.append(self.transform(image=np.array(black_img))['image'])

            return torch.stack(images)
        except:
            images = []
            while len(images) < 1: # TODO correct when faceimage are there
                black_img = Image.new('RGB', (160, 160))
                images.append(self.transform(image=np.array(black_img))['image'])
            print(f"No image found for index: {index}")
            return torch.stack(images)

    def process_audio(self, filename):
        audio_file_path = f"{self.data_folder}audio/{str(int(filename)).zfill(5)}.mp3"
        try:
            audio_data, sr = sf.read(audio_file_path)
            if sr != 16000:
                print(audio_file_path)
                raise ValueError
        except Exception as e:
            print(f"Error processing audio file {audio_file_path}: {e}")
            audio_data = np.zeros((128,), dtype=np.float32)

        return audio_data[:12*sr]

    def __len__(self):
        return len(self.label_file)
        #return int(len(self.label_file)/3)

    def collate_fn(self, batch):
        audio_data, vision_data, labels_data = zip(*batch)
        audio_data_padded = self.processor(audio_data, padding=True, sampling_rate=16000, return_tensors="pt",
                                           truncation=True, max_length=12 * 16000, return_attention_mask=True)
        lengths, permutation = audio_data_padded['attention_mask'].sum(axis=1).sort(descending=True)
        audio_packed = pack_padded_sequence(audio_data_padded['input_values'][permutation], lengths.cpu().numpy(),
                                            batch_first=True)
        # assumption: audio lengths match vision lengths; it does not hold.
        vision_data = [self.processor_vision(x, return_tensors='pt')['pixel_values'] for x in vision_data]
        vision_packed = pack_sequence([vision_data[x] for x in permutation], enforce_sorted=False)

        labels_stacked = torch.stack([labels_data[x] for x in permutation])

        return audio_packed, vision_packed, labels_stacked
