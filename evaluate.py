import numpy as np
import argparse
import spacy
import configparser

import torch
import torchtext

import datasets
import models


config = configparser.ConfigParser()
spacy_en = spacy.load('en')


def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def evaluate(sen_encoder, img_encoder, dataloader, device):
    s_list = []
    s_ids = []
    i_list = []
    i_ids = []

    with torch.no_grad():
        for imgs, sens, lengths, img_ids, ids in dataloader:
            imgs = imgs.to(device)
            img_embedded = img_encoder(imgs).to(torch.device("cpu"))
            sens = torch.transpose(sens, 0, 1)
            sens = sens.to(device)
            sen_embedded = sen_encoder(sens, lengths).to(torch.device("cpu"))

            i_list.append(img_embedded)
            s_list.append(sen_embedded)
            i_ids.append(img_ids)
            s_ids.append(ids)

    s_vectors = torch.cat(tuple(s_list)).numpy()
    s_ids = torch.cat(tuple(s_ids)).numpy()
    i_vectors = torch.cat(tuple(i_list)).numpy()
    i_ids = torch.cat(tuple(i_ids)).numpy()

    used_ids = set()
    mask = []
    for i, id in enumerate(i_ids):
        if id not in used_ids:
            used_ids.add(id)
            mask.append(True)
        else:
            mask.append(False)
    mask = np.array(mask, dtype=bool)

    s_vectors = s_vectors[mask]
    i_vectors = i_vectors[mask]

    sim_mat = np.dot(s_vectors, i_vectors.T)

    s2i, i2s = calc_retrieval_score(sim_mat)
    return s2i, i2s


def calc_retrieval_score(sim_mat):
    ks = [5, 10, 20]
    s2i = {"recall": {}, "precision": {}}
    i2s = {"recall": {}, "precision": {}}
    for k in ks:
        s2i["recall"][k] = 0.0
        s2i["precision"][k] = 0.0
        i2s["recall"][k] = 0.0
        i2s["precision"][k] = 0.0

    # image retrieval
    for i in range(sim_mat.shape[0]):
        ordered_ids = np.argsort(sim_mat[i])[::-1]
        for k in ks:
            s2i["recall"][k] += recall_at_k(i, ordered_ids, k=k)
            # s2i["precision"][k] += precision_at_k(s_ids[n], ordered_i_ids, k=k)
    for k in ks:
        s2i["recall"][k] = s2i["recall"][k] / float(sim_mat.shape[0]) * 100
        # s2i["precision"][k] = s2i["precision"][k] / float(sim_mat.shape[0]) * 100

    # sentence retrieval
    for j in range(sim_mat.shape[1]):
        ordered_ids = np.argsort(sim_mat[:, j])[::-1]
        for k in ks:
            i2s["recall"][k] += recall_at_k(j, ordered_ids, k=k)
            # i2s["precision"][k] += precision_at_k(i_ids[m], ordered_s_ids, k=k)
    for k in ks:
        i2s["recall"][k] = i2s["recall"][k] / float(sim_mat.shape[1]) * 100
        # i2s["precision"][k] = i2s["precision"][k] / float(sim_mat.shape[1]) * 100

    return s2i, i2s


def recall_at_k(gt_id, ordered_ids, k):
    TP = ordered_ids[:k].tolist().count(gt_id)
    TP_plus_FN = ordered_ids.tolist().count(gt_id)
    return float(TP) / float(TP_plus_FN)


def precision_at_k(gt_id, ordered_ids, k):
    TP = ordered_ids[:k].tolist().count(gt_id)
    return float(TP) / float(k)


def main(args):
    gpu = args.gpu
    config_path = args.config
    word2vec_path = args.word2vec
    img2vec_path = args.img2vec
    val_json_path = args.val_json
    sentence_encoder_path = args.sentence_encoder
    image_encoder_path = args.image_encoder
    name = args.name

    print("[args] gpu=%d" % gpu)
    print("[args] config_path=%s" % config_path)
    print("[args] word2vec_path=%s" % word2vec_path)
    print("[args] img2vec_path=%s" % img2vec_path)
    print("[args] val_json_path=%s" %val_json_path)
    print("[args] sentence_encoder_path=%s" % sentence_encoder_path)
    print("[args] image_encoder_path=%s" % image_encoder_path)
    print("[args] name=%s" % name)

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

    hyperparams = config["hyperparams"]
    batch_size = hyperparams.getint("batch_size")

    print("[hyperparames] batch_size=%d" % batch_size)

    # Data preparation
    word2vec = torchtext.vocab.Vectors(word2vec_path)
    dataloader_val = datasets.coco.get_loader(img2vec_path, val_json_path, word2vec, batch_size, shuffle=False, drop_last=False)

    # Model preparation
    img_encoder = models.ImageEncoder(img_size, img_hidden_size, embed_size).to(device)
    if sentence_encoder_name == 'GRU':
        sen_encoder = models.GRUEncoder(word_size, embed_size, n_layers).to(device)
    else:
        raise ValueError
    # Load params
    img_encoder.load_state_dict(torch.load(image_encoder_path))
    sen_encoder.load_state_dict(torch.load(sentence_encoder_path))
    img_encoder.eval()
    sen_encoder.eval()

    # Evaluate
    print("[info] Evaluating on the validation set ...")
    s2i, i2s = evaluate(sen_encoder, img_encoder, dataloader_val, device)
    print(
        "[validation] s2i[R@5=%.02f, R@10=%.02f, R@20=%.02f], i2s[R@5=%.02f, R@10=%.02f, R@20=%.02f]" % \
        (s2i["recall"][5], s2i["recall"][10], s2i["recall"][20],
         i2s["recall"][5], i2s["recall"][10], i2s["recall"][20]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--word2vec", type=str, default=None)
    parser.add_argument("--img2vec", type=str, default=None)
    parser.add_argument("--val_json", type=str, default=None)
    parser.add_argument("--sentence_encoder", type=str, required=True)
    parser.add_argument("--image_encoder", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)

    args = parser.parse_args()
    main(args)
