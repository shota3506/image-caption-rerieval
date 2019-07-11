import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUEncoder(nn.Module):
    def __init__(self, vocab, d_model, n_layers, dropout=0.1):
        super(GRUEncoder, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model

        self.embed = nn.Embedding.from_pretrained(vocab.vectors)
        self.gru = nn.GRU(vocab.dim, d_model, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.linear = nn.Linear(2 * n_layers * d_model, d_model)

    def forward(self, x, lengths, hidden=None):
        embedded = self.embed(x)
        embedded = torch.transpose(embedded, 0, 1)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        outputs, hidden = self.gru(packed, hidden)
        batch_size = hidden.shape[1]
        hidden = torch.transpose(hidden, 0, 1).contiguous().view(batch_size, -1)
        hidden = self.linear(hidden)
        return F.normalize(hidden)
