import numpy as np
import argparse
import configparser
import pickle

import torch
import torchvision
import torchtext

import datasets
import models
from build_vocab import Vocab


config = configparser.ConfigParser()


def encode_candidate(sen_encoder, img_encoder, dataloader, device):
    s_list = []
    s_ids = []
    i_list = []
    i_ids = []

    with torch.no_grad():
        for images, src_seq, src_pos, img_ids, ids in dataloader:
            images = images.to(device)
            src_seq = src_seq.to(device)
            src_pos = src_pos.to(device)
            img_embedded = img_encoder(images).to(torch.device("cpu"))
            sen_embedded = sen_encoder(src_seq, src_pos).to(torch.device("cpu"))

            i_list.append(img_embedded)
            s_list.append(sen_embedded)
            i_ids.append(img_ids)
            s_ids.append(ids)

    s_vectors = torch.cat(tuple(s_list))
    s_ids = torch.cat(tuple(s_ids))
    i_vectors = torch.cat(tuple(i_list))
    i_ids = torch.cat(tuple(i_ids))

    used_ids = set()
    mask = []
    for i, id in enumerate(i_ids):
        id = id.item()
        if id not in used_ids:
            used_ids.add(id)
            mask.append(True)
        else:
            mask.append(False)
    mask = torch.tensor(mask)

    i_vectors = i_vectors[mask]
    i_ids = i_ids[mask]

    return s_vectors.numpy(), s_ids.numpy(), i_vectors.numpy(), i_ids.numpy()


def main(args):
    gpu = args.gpu
    config_path = args.config
    vocab_path = args.vocab
    img2vec_path = args.img2vec
    val_json_path = args.val_json
    sentence_encoder_path = args.sentence_encoder
    image_encoder_path = args.image_encoder
    name = args.name
    mode = args.mode

    print("[args] gpu=%d" % gpu)
    print("[args] config_path=%s" % config_path)
    print("[args] word2vec_path=%s" % vocab_path)
    print("[args] img2vec_path=%s" % img2vec_path)
    print("[args] val_json_path=%s" %val_json_path)
    print("[args] sentence_encoder_path=%s" % sentence_encoder_path)
    print("[args] image_encoder_path=%s" % image_encoder_path)
    print("[args] name=%s" % name)
    print("[args] mode=%s" % mode)
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
    d_img_hidden = modelparams.getint("d_img_hidden")
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
    print("[modelparames] d_img_hidden=%d" % d_img_hidden)
    print("[modelparames] d_model=%d" % d_model)
    print()

    hyperparams = config["hyperparams"]
    batch_size = hyperparams.getint("batch_size")

    print("[hyperparames] batch_size=%d" % batch_size)
    print()

    # Data preparation
    print("[info] Loading vocabulary ...")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    dataloader_val = datasets.coco.get_loader(img2vec_path, val_json_path, vocab, batch_size)

    # Model preparation
    img_encoder = models.ImageEncoder(d_img, d_img_hidden, d_model).to(device)
    sen_encoder = models.SentenceEncoder(vocab, sentence_encoder_name, d_model, n_layers, n_head, d_k, d_v, d_inner).to(device)

    # Load params
    img_encoder.load_state_dict(torch.load(image_encoder_path))
    sen_encoder.load_state_dict(torch.load(sentence_encoder_path), strict=False)
    img_encoder.eval()
    sen_encoder.eval()

    # Evaluate
    print("[info] Encoding candidate ...")
    s_vectors, s_ids, i_vectors, i_ids = encode_candidate(sen_encoder, img_encoder, dataloader_val, device)

    if mode == 's2i':
        print('[info] Retrieving image')
        caption_id = input("input caption id: ")
        caption_id = int(caption_id)
        coco = dataloader_val.dataset.coco
        print("Caption: %s" % coco.anns[caption_id]['caption'])
        target = s_vectors[s_ids == caption_id]
        target = target.flatten()

        scores = i_vectors.dot(target)
        sorted_ids = i_ids[np.argsort(scores)[::-1]]
        for i in range(9):
            print(sorted_ids[i])

    elif mode == 'i2s':
        print('[info] Retrieving caption')
        image_id = input("input image id: ")
        image_id = int(image_id)
        coco = dataloader_val.dataset.coco
        target = i_vectors[i_ids == image_id]
        target = target.flatten()

        scores = s_vectors.dot(target)
        sorted_ids = s_ids[np.argsort(scores)[::-1]]
        for i in range(9):
            print("Caption: %s" % coco.anns[sorted_ids[i]]['caption'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--vocab", type=str, default=None)
    parser.add_argument("--img2vec", type=str, default=None)
    parser.add_argument("--val_json", type=str, default=None)
    parser.add_argument("--sentence_encoder", type=str, required=True)
    parser.add_argument("--image_encoder", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)

    args = parser.parse_args()
    main(args)
