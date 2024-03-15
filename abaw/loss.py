import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn
from torchmetrics.functional.regression import pearson_corrcoef

class MSECCC(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_function = torch.nn.MSELoss()

    def forward(self, features, labels):
        return (self.loss_function(features, labels) + (2 * torch.cov(torch.cat([features, labels], dim=1)) / (
                    features.var() + labels.var() + (features.mean() - labels.mean()) ** 2)) / 2).mean()


class CCC(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_function = torch.nn.MSELoss()

    def forward(self, features, labels):
        return 2 * torch.cov(torch.cat([features, labels], dim=1)) / (
                    features.var() + labels.var() + (features.mean() - labels.mean()) ** 2)


class MSE(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_function = torch.nn.MSELoss()

    def forward(self, features, labels):
        return self.loss_function(features, labels)

class CORR(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function = pearson_corrcoef

    def forward(self, predictions, labels):
        return (1 - torch.nan_to_num(self.loss_function(predictions, labels), nan=-1.0)).mean()
