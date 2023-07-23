### https://github.com/dydjw9/Efficient_SAM/blob/main/utils/cutout.py

import torch
import torch.nn as nn


class Cutout(nn.Module):
    def __init__(self, size=16, p=0.5, training=True, same_batch_same_result=False):
        super().__init__()
        self.size = size
        self.half_size = size // 2
        self.p = p
        self.training = training
        self.same_batch_same_result = same_batch_same_result

    def forward(self, input):
        if not self.training:
            return input

        if torch.rand([1]).item() > self.p:
            return input

        new_input = input.clone()
        if self.same_batch_same_result:
            torch.random.manual_seed(id(input))

        left = torch.randint(-self.half_size, new_input.size(1) - self.half_size, [1]).item()
        top = torch.randint(-self.half_size, new_input.size(2) - self.half_size, [1]).item()
        right = min(new_input.size(1), left + self.size)
        bottom = min(new_input.size(2), top + self.size)

        new_input[:, max(0, left): right, max(0, top): bottom] = 0
        return new_input
