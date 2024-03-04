import torch
import timm
import numpy as np
import torch.nn as nn


class Model(nn.Module):

    def __init__(self,
                 model_name,
                 pretrained=True,
                 img_size=383):

        super(Model, self).__init__()

        self.img_size = img_size

        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        self.model = nn.Linear(1152, 6)

    def get_config(self, ):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config

    def forward(self, vit_features, audio_features):

        return self.model(torch.cat([vit_features, audio_features], dim=1))
