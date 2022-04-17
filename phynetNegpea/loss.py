import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
tr = torch

class NegPeaLoss(nn.Module):
    def __init__(self):
        super(NegPeaLoss, self).__init__()

    def forward(self, x, y):
        if len(x.size()) == 1:
            x = tr.unsqueeze(x, 0)
            y = tr.unsqueeze(y, 0)
        T = x.shape[1]
        p_coeff = tr.sub(T * tr.sum(tr.mul(x, y), 1), tr.mul(tr.sum(x, 1), tr.sum(y, 1)))
        norm = tr.sqrt((T * tr.sum(x ** 2, 1) - tr.sum(x, 1) ** 2) * (T * tr.sum(y ** 2, 1) - tr.sum(y, 1) ** 2))
        p_coeff = tr.div(p_coeff, norm)
        losses = tr.tensor(1.) - p_coeff
        totloss = tr.mean(losses)
        return totloss
    
# if __name__ == "__main__":

#     crit = NegPeaLoss()

#     outputs = tr.randn(128)
#     targets = tr.randn(1)

#     print(NegPeaLoss(outputs, targets))