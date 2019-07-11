import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxPoolingEncoder(nn.Module):
    def __init__(self, vocab, d_model):
        super(MaxPoolingEncoder, self).__init__()
        self.d_model = d_model

        self.embed = nn.Embedding.from_pretrained(vocab.vectors)
        self.linear = nn.Linear(vocab.dim, d_model)

    def forward(self, x):
        embedded = self.embed(x)
        maxpooled, _ = torch.max(embedded, dim=1)
        hidden = self.linear(maxpooled)
        return F.normalize(hidden)
