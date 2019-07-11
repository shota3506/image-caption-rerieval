import numpy as np
import argparse
import configparser
import pickle

import torch
import torchtext

import datasets
import models
from train import encode_image, encode_sentence
from build_vocab import Vocab


config = configparser.ConfigParser()


def evaluate(sen_encoder, img_encoder, dataloader, device):
    s_list = []
    s_ids = []
    i_list = []
    i_ids = []

    with torch.no_grad():
        for imgs, sens, lengths, img_ids, ids in dataloader:
            img_embedded = encode_image(img_encoder, imgs, device).to(torch.device("cpu"))
            sen_embedded = encode_sentence(sen_encoder, sens, lengths, device).to(torch.device("cpu"))

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
    vocab_path = args.vocab
    img2vec_path = args.img2vec
    val_json_path = args.val_json
    sentence_encoder_path = args.sentence_encoder
    image_encoder_path = args.image_encoder
    name = args.name

    print("[args] gpu=%d" % gpu)
    print("[args] config_path=%s" % config_path)
    print("[args] word2vec_path=%s" % vocab_path)
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

    hyperparams = config["hyperparams"]
    batch_size = hyperparams.getint("batch_size")

    print("[hyperparames] batch_size=%d" % batch_size)

    print("[info] Loading vocabulary ...")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    dataloader_val = datasets.coco.get_loader(img2vec_path, val_json_path, vocab, batch_size)

    # Model preparation
    img_encoder = models.ImageEncoder(d_img, d_img_hidden, d_model).to(device)
    if sentence_encoder_name == 'GRU':
        sen_encoder = models.GRUEncoder(vocab, d_model, n_layers).to(device)
    elif sentence_encoder_name == 'LSTM':
        sen_encoder = models.LSTMEncoder(vocab, d_model, n_layers).to(device)
    elif sentence_encoder_name == 'Transformer':
        sen_encoder = models.TransformerEncoder(vocab, n_layers, n_head, d_k, d_v, d_model, d_inner).to(device)
    elif sentence_encoder_name == 'MaxPooling':
        sen_encoder = models.MaxPoolingEncoder(vocab, d_model).to(device)
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
    parser.add_argument("--vocab", type=str, default=None)
    parser.add_argument("--img2vec", type=str, default=None)
    parser.add_argument("--val_json", type=str, default=None)
    parser.add_argument("--sentence_encoder", type=str, required=True)
    parser.add_argument("--image_encoder", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)

    args = parser.parse_args()
    main(args)
