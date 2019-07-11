import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):
    def __init__(self, d_word, d_model, n_layers, dropout=0.1):
        super(GRU, self).__init__()
        self.n_layers = n_layers
        self.d_word = d_word
        self.d_model = d_model

        self.gru = nn.GRU(d_word, d_model, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.linear = nn.Linear(2 * n_layers * d_model, d_model)

    def forward(self, enc_seq, src_pos, hidden=None):
        enc_seq = torch.transpose(enc_seq, 0, 1)
        lengths, _ = torch.max(src_pos, dim=1)
        packed = nn.utils.rnn.pack_padded_sequence(enc_seq, lengths)
        outputs, hidden = self.gru(packed, hidden)
        batch_size = hidden.shape[1]
        hidden = torch.transpose(hidden, 0, 1).contiguous().view(batch_size, -1)
        return self.linear(hidden)
