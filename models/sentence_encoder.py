import torch
import torch.nn as nn
import torch.nn.functional as F
from .gru import GRU
from .lstm import LSTM
from .transformer import TransformerEncoder
from .max_polling import MaxPooling


class SentenceEncoder(nn.Module):

    def __init__(self, vocab, encoder_name, d_model, n_layers=None, n_head=None, d_k=None, d_v=None, d_inner=None):
        super(SentenceEncoder, self).__init__()

        self.embed = nn.Embedding.from_pretrained(vocab.vectors)
        self.d_word = self.embed.embedding_dim
        self.encoder_name = encoder_name

        if encoder_name == 'GRU':
            assert n_layers
            self.encoder = GRU(self.d_word, d_model, n_layers)
        elif encoder_name == 'LSTM':
            assert n_layers
            self.encoder = LSTM(self.d_word, d_model, n_layers)
        elif encoder_name == 'Transformer':
            assert n_layers and n_head and d_k and d_v and d_inner
            self.encoder = TransformerEncoder(self.d_word, d_model, n_layers, n_head, d_k, d_v, d_inner)
        elif encoder_name == 'MaxPooling':
            self.encoder = MaxPooling(self.d_word, d_model)
        else:
            raise ValueError

    def forward(self, src_seq, src_pos):
        enc_seq = self.embed(src_seq)
        out = self.encoder(enc_seq, src_pos)
        return F.normalize(out)
