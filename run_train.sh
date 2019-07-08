#!/usr/bin/env sh

GPU=0
CONFIG=./config/trial1.ini
WORD2VEC=$HOME/dataset/glove/glove.840B.300d.txt
IMG2VEC=./features/train2017.resnet50.2048d.pth
TRAIN_JSON=$HOME/dataset/coco/annotations/captions_train2017.json
SAVE=./save
NAME=trial1

python train.py \
    --gpu ${GPU} \
    --config ${CONFIG} \
    --word2vec ${WORD2VEC} \
    --img2vec ${IMG2VEC} \
    --train_json ${TRAIN_JSON} \
    --save ${SAVE} \
    --name ${NAME}
