import torch
import torchtext
import os
import pickle
import argparse
from tqdm import tqdm


class Vocab(object):
    """Simple vocabulary wrapper."""
    def __init__(self, pretrained):
        pretrained_vectors = torchtext.vocab.Vectors(pretrained)
        self.dim = pretrained_vectors.dim
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.stoi = {self.pad_token: 0, self.unk_token: 1}
        self.itos = {0: self.pad_token, 1: self.unk_token}
        self.vectors = torch.zeros(len(self.stoi), self.dim)
        self.load_vectors(pretrained_vectors)

    def add_word(self, word):
        if word not in self.stoi:
            idx = len(self.stoi)
            self.stoi[word] = idx
            self.itos[idx] = word
            return True
        else:
            return False

    def load_vectors(self, pretrained_vectors):
        mask = []
        for s in tqdm(pretrained_vectors.itos):
            mask.append(self.add_word(s))
        mask = torch.tensor(mask)
        self.vectors = torch.cat((self.vectors, pretrained_vectors.vectors[mask, :]), dim=0)

    def __call__(self, word):
        if word not in self.stoi:
            return self.stoi[self.unk_token]
        return self.stoi[word]

    def __len__(self):
        return len(self.stoi)


def build_vocab(pretrained_path):
    """Build a simple vocabulary wrapper."""
    vocab = Vocab(pretrained_path)
    return vocab


def main(args):
    save_path = args.save_path
    pretrained_path = args.pretrained_path
    name = args.name

    print("[args] save_path=%s" % save_path)
    print("[args] pretrained_path=%s" % pretrained_path)
    print("[args] name=%s" % name)

    vocab = build_vocab(pretrained_path)
    with open(os.path.join(save_path, name + ".vocab.pkl"), 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(os.path.join(save_path, name + ".vocab.pkl")))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, required=True,
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--pretrained_path', type=str, default='None')
    parser.add_argument('--name', type=str, required=True)

    args = parser.parse_args()
    main(args)
