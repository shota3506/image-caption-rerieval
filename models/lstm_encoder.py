import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self):
        super(LSTMEncoder, self).__init__()

    def forward(self, x):
        print('hoge')
