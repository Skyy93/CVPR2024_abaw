import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn


class MSECCC(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_function = torch.nn.MSELoss()

    def forward(self, features, labels):
        return (self.loss_function(features, labels) + 2 * torch.cov(features, labels) / (
                    features.var() + labels.var() + (features.mean() - labels.mean()) ** 2)) / 2


class CCC(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_function = torch.nn.MSELoss()

    def forward(self, features, labels):
        return 2 * torch.cov(features, labels) / (
                    features.var() + labels.var() + (features.mean() - labels.mean()) ** 2)


class MSE(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_function = torch.nn.MSELoss()

    def forward(self, features, labels):
        return self.loss_function(features, labels)
