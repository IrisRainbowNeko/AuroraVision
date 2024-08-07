import torch
from torch import nn


class EntropyLoss(nn.Module):
    '''
    降低输出的信息熵，让每个token尽可能预测不同的东西
    '''
    def __init__(self, T=1., eps=1e-8):
        super().__init__()
        self.T = T
        self.eps = eps

    def entropy(self, x, dim=0):
        probs = torch.softmax(x*self.T, dim=dim)
        return -torch.sum(probs * torch.log(probs + self.eps), dim=dim).mean()

    def forward(self, x):
        row_entropy = self.entropy(x, dim=-2)
        col_entropy = self.entropy(x, dim=-1)

        return row_entropy + col_entropy
