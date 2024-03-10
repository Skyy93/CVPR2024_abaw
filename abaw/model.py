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
            self.vision_model = timm.create_model(model_name[0], pretrained=True, num_classes=0)
            self.audio_model = Wav2Vec2BertModel.from_pretrained(model_name[1])
            self.fusion_model = nn.Linear(3072,6)#Mamba.from_pretrained('state-spaces/mamba-130m')
            self.lstm_audio = nn.LSTM(1024, 512, num_layers=1, batch_first=True, bidirectional=False)
            self.lstm_vision = nn.LSTM(1024, 512, num_layers=1, batch_first=True, bidirectional=False)
            self.pooling_transformer = nn.TransformerEncoderLayer(d_model=1024, nhead=16, dim_feedforward=2048)

    def forward(self, audio, vision):
        if self.linear:
            return self.model(torch.cat([torch.mean(vision, dim=0), torch.mean(audio, dim=0)], dim=1))
        else:
            with torch.no_grad():
                vision_outputs = []

                for i in range(vision.size(1)):
                    frame = vision[:, i, :, :, :] 
                    vision_embeddings = self.vision_model(frame) 
                    vision_outputs.append(vision_embeddings)    
                vision_outputs = torch.stack(vision_outputs, dim=1)  # Stack along new sequence dimension
                audio_output = self.audio_model(**audio).last_hidden_state
                pooled_audio = audio_output.mean(1)
            

            lstm_audio = self.lstm_audio(audio_output)
            lstm_vision = self.lstm_vision(vision_outputs)
            pooled_vision = self.pooling_transformer(vision_outputs)
            pooled_vision = pooled_vision.mean(1)  # Mean pooling across the sequence dimension
            fusion_input = torch.cat([pooled_vision, lstm_vision[0][:, -1, :], pooled_audio, lstm_audio[0][:, -1, :]], dim=1)
            pred = self.fusion_model(fusion_input)

            return pred
