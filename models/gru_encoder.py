import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout=0.1):
        super(GRUEncoder, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.linear = nn.Linear(2 * n_layers * hidden_size, hidden_size)

    def forward(self, embedded, input_lengths, hidden=None):
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        batch_size = hidden.shape[1]
        hidden = torch.transpose(hidden, 0, 1).contiguous().view(batch_size, -1)
        hidden = self.linear(hidden)
        return F.normalize(hidden)
