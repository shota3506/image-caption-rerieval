#!/usr/bin/env sh

GPU=0
MODEL=resnet50
ROOT=$HOME/dataset/coco/train2017
JSON=$HOME/dataset/coco/annotations/captions_train2017.json
SAVE=./features
DATA=train2017
BATCH_SIZE=512

python prepare.py \
    --gpu ${GPU} \
    --model ${MODEL} \
    --root ${ROOT} \
    --json ${JSON} \
    --save ${SAVE} \
    --data ${DATA} \
    --batch_size ${BATCH_SIZE}
