import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelFiltering(nn.Module):
    def __init__(self, model, idx):
        super().__init__()
        self.model = model
        self.idx = idx

    def forward(self, input):
        return self.model(input)[:, self.idx]
