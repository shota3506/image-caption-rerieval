import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):

    def __init__(self, d_input, d_hidden, d_model):
        super(ImageEncoder, self).__init__()

        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_model = d_model

        self.encoder = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, x):
        return F.normalize(self.encoder(x))
