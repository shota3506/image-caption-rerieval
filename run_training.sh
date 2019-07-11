#!/usr/bin/env sh

GPU=0
CONFIG=./config/trial1.ini
VOCAB=./vocab/glove.840B.300d.vocab.pkl
IMG2VEC=./features/train2017.resnet50.2048d.pth
TRAIN_JSON=$HOME/dataset/coco/annotations/captions_train2017.json
SAVE=./save
NAME=trial1

python train.py \
    --gpu ${GPU} \
    --config ${CONFIG} \
    --vocab ${VOCAB} \
    --img2vec ${IMG2VEC} \
    --train_json ${TRAIN_JSON} \
    --save ${SAVE} \
    --name ${NAME}
