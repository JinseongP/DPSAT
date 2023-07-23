### https://github.com/dydjw9/Efficient_SAM/blob/main/utils/cutout.py
import torch
import torch.nn as nn


class Gaussian(nn.Module):
    def __init__(self, std=0.01, p=0.5, training=True, same_batch_same_result=False):
        super().__init__()
        self.std = std
        self.p = p
        self.training = training
        self.same_batch_same_result = same_batch_same_result

    def forward(self, input):
        if not self.training:
            return input

        if torch.rand([1]).item() > self.p:
            return input

        new_input = input.clone()
        if self.same_batch_for_same_result:
            torch.random.manual_seed(id(input))
        new_input = input + torch.randn_like(input)*self.std
        return new_input
