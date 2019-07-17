import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):

    def __init__(self, d_input, d_model):
        super(ImageEncoder, self).__init__()

        self.d_input = d_input
        self.d_model = d_model
        self.encoder = nn.Linear(d_input, d_model)

    def forward(self, x):
        return F.normalize(self.encoder(x))
