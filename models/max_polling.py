import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxPooling(nn.Module):
    def __init__(self, d_word, d_model):
        super(MaxPooling, self).__init__()
        self.d_word = d_word
        self.d_model = d_model

        self.linear = nn.Linear(d_word, d_model)

    def forward(self, enc_seq, src_pos):
        maxpooled, _ = torch.max(enc_seq, dim=1)
        return self.linear(maxpooled)
