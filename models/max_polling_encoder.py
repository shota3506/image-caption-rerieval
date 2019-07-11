import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxPoolingEncoder(nn.Module):
    def __init__(self, vocab, hidden_size):
        super(MaxPoolingEncoder, self).__init__()
        self.hidden_size = hidden_size

        self.embed = nn.Embedding.from_pretrained(vocab.vectors)
        self.linear = nn.Linear(vocab.dim, hidden_size)

    def forward(self, x):
        embedded = self.embed(x)
        maxpooled, _ = torch.max(embedded, dim=1)
        hidden = self.linear(maxpooled)
        return F.normalize(hidden)
