import torch
import timm
import numpy as np
import torch.nn as nn
from transformers import Wav2Vec2BertModel, Wav2Vec2Model, ViTForImageClassification
from abaw.mamba import Mamba
from torch.nn.utils.rnn import unpack_sequence, pack_sequence
from abaw.audeer import EmotionModel

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
            self.vision_model = ViTForImageClassification.from_pretrained(model_name[0], output_hidden_states=True)


            #self.audio_model = Wav2Vec2BertModel.from_pretrained(model_name[1])
            #self.audio_model = Wav2Vec2Model.from_pretrained(model_name[1])
            #self.audio_model = EmotionModel.from_pretrained(model_name[1])
            #self.fusion_model = nn.Linear(4102,6)#Mamba.from_pretrained('state-spaces/mamba-130m')
            self.fusion_model = nn.Sequential(nn.Linear(2*775, 2*775),
                                              nn.Tanh(),
                                              nn.Linear(2*775, 6))
            #self.lstm_audio = nn.LSTM(1027, 1027, num_layers=2, batch_first=True, bidirectional=False)
            self.lstm_vision = nn.LSTM(775, 775, num_layers=2, batch_first=True, bidirectional=False)

            #self.pooling_transformer = nn.TransformerEncoderLayer(d_model=1024, nhead=16, dim_feedforward=2048)

    def forward(self, audio, vision):
        if self.linear:
            return self.model(torch.cat([torch.mean(vision, dim=0), torch.mean(audio, dim=0)], dim=1))
        else:
            #with torch.no_grad():
            vision = [self.vision_model(x) for x in unpack_sequence(vision)]
            vision = [torch.concat([torch.stack(x.hidden_states).mean((2, 0)), x.logits], dim=-1) for x in vision]

            #for i in range(vision.size(1)):
            #    frame = vision[:, i, :, :, :]
            #    vision_embeddings = self.vision_model(frame)
            #    vision_outputs.append(vision_embeddings)
            #vision_outputs = torch.stack(vision_outputs, dim=1)  # Stack along new sequence dimension
            #vision_outputs = torch.nn.utils.rnn.pack_sequence()

            #audio = [self.audio_model(x[None, :]).last_hidden_state[0] for x in unpack_sequence(audio)]
            #audio = [self.audio_model(x[None, :]) for x in unpack_sequence(audio)]
            #audio = [torch.hstack([x.squeeze(), y.squeeze()]) for x, y in audio]

            #pooled_audio = torch.stack([x.mean(0) for x in audio])
            pooled_vision = torch.stack([x.mean(0) for x in vision])
            

            #lstm_audio, _ = self.lstm_audio(pack_sequence(audio))
            lstm_vision, _ = self.lstm_vision(pack_sequence(vision, enforce_sorted=False))

            #pooled_vision = self.pooling_transformer(vision_outputs)
            #pooled_vision = pooled_vision.mean(1)  # Mean pooling across the sequence dimension

            #fusion_input = torch.cat([pooled_vision, torch.stack([x[-1, :] for x in unpack_sequence(lstm_vision)]), pooled_audio, torch.stack([x[-1, :] for x in unpack_sequence(lstm_audio)])], dim=1)
            #fusion_input = torch.cat([pooled_audio, torch.stack([x[-1, :] for x in unpack_sequence(lstm_audio)])], dim=1)
            fusion_input = torch.cat([pooled_vision, torch.stack([x[-1, :] for x in unpack_sequence(lstm_vision)])],
                                     dim=1)

            pred = self.fusion_model(fusion_input)

            return pred
