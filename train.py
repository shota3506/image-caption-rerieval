import argparse
import os
import configparser
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchtext

import datasets
import models
from build_vocab import Vocab


config = configparser.ConfigParser()


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
    vocab_path = args.vocab
    img2vec_path = args.img2vec
    train_json_path = args.train_json
    name = args.name
    save_path = args.save

    print("[args] gpu=%d" % gpu)
    print("[args] config_path=%s" % config_path)
    print("[args] word2vec_path=%s" % vocab_path)
    print("[args] img2vec_path=%s" % img2vec_path)
    print("[args] train_json_path=%s" % train_json_path)
    print("[args] name=%s" % name)
    print("[args] save_path=%s" % save_path)
    print()

    device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")

    config.read(config_path)

    # Model parameters
    modelparams = config["modelparams"]
    sentence_encoder_name = modelparams.get("sentence_encoder")
    n_layers = modelparams.getint("n_layers")
    n_head = modelparams.getint("n_head")
    d_k = modelparams.getint("d_k")
    d_v = modelparams.getint("d_v")
    d_inner = modelparams.getint("d_inner")
    d_img = modelparams.getint("d_img")
    d_model = modelparams.getint("d_model")

    print("[modelparames] sentence_encoder_name=%s" % sentence_encoder_name)
    if n_layers:
        print("[modelparames] n_layers=%d" % n_layers)
    if n_head:
        print("[modelparames] n_head=%d" % n_head)
    if d_k:
        print("[modelparames] d_k=%d" % d_k)
    if d_v:
        print("[modelparames] d_v=%d" % d_v)
    if d_inner:
        print("[modelparames] d_inner=%d" % d_inner)
    print("[modelparames] d_img=%d" % d_img)
    print("[modelparames] d_model=%d" % d_model)
    print()

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
    print()

    # Data preparation
    print("[info] Loading vocabulary ...")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    dataloader_train = datasets.coco.get_loader(img2vec_path, train_json_path, vocab, batch_size)

    # Model preparation
    img_encoder = models.ImageEncoder(d_img, d_model).to(device)
    sen_encoder = models.SentenceEncoder(vocab, sentence_encoder_name, d_model, n_layers, n_head, d_k, d_v, d_inner).to(device)

    img_optimizer = optim.Adam(img_encoder.parameters(), lr=lr, weight_decay=weight_decay)
    sen_optimizer = optim.Adam(sen_encoder.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = PairwiseRankingLoss(margin=margin)

    # Train
    print("[info] Training ...")
    for epoch in range(n_epochs):
        pbar = tqdm(dataloader_train)
        running_loss = 0.0

        for i, (images, src_seq, src_pos, _, _) in enumerate(pbar):
            pbar.set_description('epoch %3d / %d' % (epoch + 1, n_epochs))

            images = images.to(device)
            src_seq = src_seq.to(device)
            src_pos = src_pos.to(device)
            img_embedded = img_encoder(images)
            sen_embedded = sen_encoder(src_seq, src_pos)

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
            if (i + 1) % args.print_every == 0:
                pbar.set_postfix(loss=running_loss / args.print_every)
                running_loss = 0

        if (epoch + 1) % args.save_every == 0:
            save_dir = os.path.join(save_path, name)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            sen_dict = sen_encoder.state_dict()
            sen_dict.pop('embed.weight')
            img_dict = img_encoder.state_dict()
            torch.save(sen_dict, os.path.join(
                save_dir, 'sentence_encoder-{}.pth'.format(epoch + 1)))
            torch.save(img_dict, os.path.join(
                save_dir, 'image_encoder-{}.pth'.format(epoch + 1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--vocab", type=str, default=None)
    parser.add_argument("--img2vec", type=str, default=None)
    parser.add_argument("--train_json", type=str, default=None)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--print_every", type=int, default=300)
    parser.add_argument("--save_every", type=int, default=5)

    args = parser.parse_args()
    main(args)
