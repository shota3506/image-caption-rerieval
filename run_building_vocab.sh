#!/usr/bin/env sh

SAVE=./vocab
PRETRAINED=$HOME/dataset/glove/glove.840B.300d.txt
NAME=glove.840B.300d

python build_vocab.py \
    --save_path ${SAVE} \
    --pretrained_path ${PRETRAINED} \
    --name ${NAME}