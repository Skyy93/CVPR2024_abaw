import torch
import timm
import numpy as np
import torch.nn as nn
from transformers import Wav2Vec2BertModel
from abaw.mamba import Mamba

class Model(nn.Module):

    def __init__(self,
                 model_name,
                 ):

        super(Model, self).__init__()
        self.linear = False
        if "linear" in model_name[0]:
            self.model = nn.Linear(1152, 6)
            self.linear = True
        else:
            #self.vision_model = timm.create_model(model_name[0], pretrained=True, num_classes=0)
            self.audio_model = Wav2Vec2BertModel.from_pretrained(model_name[1])
            self.fusion_model = nn.Linear(1024,6)#Mamba.from_pretrained('state-spaces/mamba-130m')


    def forward(self, audio, vision):
        if self.linear:
            return self.model(torch.cat([torch.mean(vision, dim=0), torch.mean(audio, dim=0)], dim=1))
        else:
            with torch.no_grad():
                #vision = self.vision_model(**vision)
                audio = self.audio_model(**audio).last_hidden_state
                audio = audio.mean(1)
            pred = self.fusion_model(audio)
            return pred


