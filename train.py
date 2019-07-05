import argparse
import os
import time
import spacy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchtext

import datasets
import models


spacy_en = spacy.load('en')


def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def main(args):
    gpu = args.gpu
    model_name = args.model
    word2vec_path = args.word2vec
    img2vec_path = args.img2vec
    train_json_path = args.train_json
    trial_name = args.name

    print("[args] gpu=%d" % gpu)
    print("[args] model_name=%s" % model_name)
    print("[args] word2vec_path=%s" % word2vec_path)
    print("[args] img2vec_path=%s" % img2vec_path)
    print("[args] train_json_path=%s" % train_json_path)
    print("[args] trial_name=%s" % trial_name)

    # Hyper parameters
    word_size = 300
    img_size = 2048
    img_hidden_size = 2048
    embed_size = 1024
    n_layers = 2
    margin = 0.1
    weight_decay = 0.00001
    grad_clip = 5.0
    lr = 0.001
    # batch_size = 256
    batch_size = 2

    # Data preparation
    word2vec = torchtext.vocab.Vectors(word2vec_path)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataloader_train = datasets.coco.get_loader(img2vec_path, train_json_path, word2vec, transform, batch_size, True, 1)

    # Model preparation
    img_encoder = models.ImageEncoder(img_size, img_hidden_size, embed_size)
    seq_encoder = models.GRUEncoder(word_size, embed_size, n_layers)

    img_optimizer = optim.Adam(img_encoder.parameters(), lr=lr, weight_decay=weight_decay)
    sen_optimizer = optim.Adam(seq_encoder.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.MSELoss()

    for epoch in range(1):
        for imgs, seqs, lengths in dataloader_train:
            img_embedded = img_encoder(imgs)

            seqs = torch.transpose(seqs, 0, 1)
            seq_output, seq_hidden = seq_encoder(seqs, lengths)
            seq_embedded = torch.mean(seq_hidden, dim=0)

            img_optimizer.zero_grad()
            sen_optimizer.zero_grad()

            loss = criterion(img_embedded, seq_embedded)
            loss.backward()

            nn.utils.clip_grad_value_(img_encoder.parameters(), grad_clip)
            nn.utils.clip_grad_value_(seq_encoder.parameters(), grad_clip)

            img_optimizer.step()
            sen_optimizer.step()

            print(loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--word2vec", type=str, default=None)
    parser.add_argument("--img2vec", type=str, default=None)
    parser.add_argument("--train_json", type=str, default=None)
    parser.add_argument("--name", type=str, required=True)

    args = parser.parse_args()
    main(args)
