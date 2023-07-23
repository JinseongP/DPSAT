## ref:https://github.com/wangleiofficial/label-smoothing-pytorch

import torch.nn.functional as F
import torch.nn as nn

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return self.linear_combination(loss / n, nll, self.epsilon)

    def linear_combination(self, x, y, epsilon):
        return epsilon * x + (1 - epsilon) * y

    def reduce_loss(self, loss, reduction='mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss
