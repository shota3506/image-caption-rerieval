# Image Caption Retrieval

This repository provides models for image caption retrieval.

### Requirement
- python 3.5+
- pytorch
- torchvision
- torchtext
- tqdm
- numpy

### Dataset 
[COCO](http://cocodataset.org/#home) for Captioning

### Sentence encoders
#### GRU
Gated Recurrent Unit encoder for sentence embedding

#### LSTM
Long Short Term Memory encoder for sentence embedding

#### Transformer
[Transformer](https://arxiv.org/abs/1706.03762) encoder unit (but this model uses pretrained word embedding)

#### Maxpooling
Max-pooling pretrained word vectors 
