#!/usr/bin/env sh

GPU=0
CONFIG=./config/trial1.ini
VOCAB=./vocab/glove.840B.300d.vocab.pkl
IMG2VEC=./features/val2017.resnet50.2048d.pth
VAL_JSON=$HOME/dataset/coco/annotations/captions_val2017.json
SENTENCE_ENCODER=./save/trial1/sentence_encoder-30.pth
IMAGE_ENCODER=./save/trial1/image_encoder-30.pth
NAME=trial1

python evaluate.py \
    --gpu ${GPU} \
    --config ${CONFIG} \
    --vocab ${VOCAB} \
    --img2vec ${IMG2VEC} \
    --val_json ${VAL_JSON} \
    --sentence_encoder ${SENTENCE_ENCODER} \
    --image_encoder ${IMAGE_ENCODER} \
    --name ${NAME}
