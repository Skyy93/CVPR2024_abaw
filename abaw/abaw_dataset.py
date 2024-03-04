import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import torch
from tqdm import tqdm
import time

cv2.setNumThreads(2)

class HumeDatasetTrain(Dataset):
    
    def __init__(self,
                 data_folder,
                 label_file=None,
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.label_file = label_file
            

    def __getitem__(self, index):

        
        return None
    
    def __len__(self):
        return len(self.samples)
        

            
       
class HumeDatasetEval(Dataset):

    def __init__(self,
                 data_folder,
                 label_file=None,
                 ):
        super().__init__()

        self.data_folder = data_folder
        self.label_file = label_file

    def __getitem__(self, index):
        return None

    def __len__(self):
        return len(self.samples)




