import torch
import torch.nn as nn
from torch.nn import init


def init_weights(m):
    if type(m) == nn.Linear:
        init.kaiming_normal_(m.weight.data)
        init.constant_(m.bias.data, val=0)


class ImageEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, embed_size):
        super(ImageEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_size)
        )

        self.encoder.apply(init_weights)

    def forward(self, x):
        return self.encoder(x)
