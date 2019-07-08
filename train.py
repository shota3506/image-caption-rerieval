import argparse
import os
import time
import spacy
import configparser
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchtext

import datasets
import models


config = configparser.ConfigParser()
spacy_en = spacy.load('en')


def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        p_product = torch.sum(anchor * positive, dim=1)
        n_product = torch.sum(anchor * negative, dim=1)
        dist_hinge = torch.clamp(self.margin - p_product + n_product, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss


def main(args):
    gpu = args.gpu
    config_path = args.config
    word2vec_path = args.word2vec
    img2vec_path = args.img2vec
    train_json_path = args.train_json
    name = args.name
    save_path = args.save

    print("[args] gpu=%d" % gpu)
    print("[args] config_path=%s" % config_path)
    print("[args] word2vec_path=%s" % word2vec_path)
    print("[args] img2vec_path=%s" % img2vec_path)
    print("[args] train_json_path=%s" % train_json_path)
    print("[args] name=%s" % name)
    print("[args] save_path=%s" % save_path)

    device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")

    config.read(config_path)

    # Model parameters
    modelparams = config["modelparams"]
    sentence_encoder_name = modelparams.get("sentence_encoder")
    n_layers = modelparams.getint("n_layers")
    word_size = modelparams.getint("word_size")
    img_size = modelparams.getint("img_size")
    img_hidden_size = modelparams.getint("img_hidden_size")
    embed_size = modelparams.getint("embed_size")

    print("[modelparames] sentence_encoder_name=%s" % sentence_encoder_name)
    print("[modelparames] n_layers=%d" % n_layers)
    print("[modelparames] word_size=%d" % word_size)
    print("[modelparames] img_size=%d" % img_size)
    print("[modelparames] img_hidden_size=%d" % img_hidden_size)
    print("[modelparames] embed_size=%d" % embed_size)

    # Hyper parameters
    hyperparams = config["hyperparams"]
    margin = hyperparams.getfloat("margin")
    weight_decay = hyperparams.getfloat("weight_decay")
    grad_clip = hyperparams.getfloat("grad_clip")
    lr = hyperparams.getfloat("lr")
    batch_size = hyperparams.getint("batch_size")
    n_epochs = hyperparams.getint("n_epochs")
    n_negatives = hyperparams.getint("n_negatives")

    print("[hyperparames] margin=%f" % margin)
    print("[hyperparames] weight_decay=%f" % weight_decay)
    print("[hyperparames] grad_clip=%f" % grad_clip)
    print("[hyperparames] lr=%f" % lr)
    print("[hyperparames] batch_size=%d" % batch_size)
    print("[hyperparames] n_epochs=%d" % n_epochs)
    print("[hyperparames] n_negatives=%d" % n_negatives)

    # Data preparation
    word2vec = torchtext.vocab.Vectors(word2vec_path)
    dataloader_train = datasets.coco.get_loader(img2vec_path, train_json_path, word2vec, batch_size)

    # Model preparation
    img_encoder = models.ImageEncoder(img_size, img_hidden_size, embed_size).to(device)
    if sentence_encoder_name == 'GRU':
        sen_encoder = models.GRUEncoder(word_size, embed_size, n_layers).to(device)
    else:
        raise ValueError

    img_optimizer = optim.Adam(img_encoder.parameters(), lr=lr, weight_decay=weight_decay)
    sen_optimizer = optim.Adam(sen_encoder.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = PairwiseRankingLoss(margin=margin)

    for epoch in range(n_epochs):
        pbar = tqdm(dataloader_train)
        running_loss = 0.0

        # Train
        for i, (imgs, sens, lengths, _, _) in enumerate(pbar):
            pbar.set_description('epoch %3d / %d' % (epoch + 1, n_epochs))

            imgs = imgs.to(device)
            img_embedded = img_encoder(imgs)

            sens = torch.transpose(sens, 0, 1)
            sens = sens.to(device)
            sen_embedded = sen_encoder(sens, lengths)

            img_optimizer.zero_grad()
            sen_optimizer.zero_grad()

            loss = 0.0
            for _ in range(n_negatives):
                perm = torch.randperm(len(img_embedded))
                img_shuffled = img_embedded[perm]
                loss += criterion(sen_embedded, img_embedded, img_shuffled)
            loss /= n_negatives
            loss.backward()

            nn.utils.clip_grad_value_(img_encoder.parameters(), grad_clip)
            nn.utils.clip_grad_value_(sen_encoder.parameters(), grad_clip)
            img_optimizer.step()
            sen_optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                pbar.set_postfix(loss=running_loss / 100)
                running_loss = 0

        if (epoch + 1) % 1 == 0:
            save_dir = os.path.join(save_path, name)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            torch.save(sen_encoder.state_dict(), os.path.join(
                save_dir, 'sentence_encoder-{}.pth'.format(epoch + 1)))
            torch.save(img_encoder.state_dict(), os.path.join(
                save_dir, 'image_encoder-{}.pth'.format(epoch + 1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--word2vec", type=str, default=None)
    parser.add_argument("--img2vec", type=str, default=None)
    parser.add_argument("--train_json", type=str, default=None)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)

    args = parser.parse_args()
    main(args)
