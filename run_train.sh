#!/usr/bin/env sh

GPU=0
MODEL=cvmcnn
WORD2VEC=$HOME/dataset/glove/glove.840B.300d.txt
IMG2VEC=./features/train2017.resnet50.2048d.pth
TRAIN_JSON=$HOME/dataset/coco/annotations/captions_train2017.json
NAME=trial1
BATCH_SIZE=64

python train.py \
    --gpu ${GPU} \
    --model ${MODEL} \
    --word2vec ${WORD2VEC} \
    --img2vec ${IMG2VEC} \
    --train_json ${TRAIN_JSON} \
    --name ${NAME} \
