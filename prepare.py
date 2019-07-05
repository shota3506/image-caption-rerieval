import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision

from PIL import Image
from pycocotools.coco import COCO


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        return x


def main(args):
    gpu = args.gpu
    model_name = args.model
    root_path = args.root
    json_path = args.json
    save_path = args.save
    data = args.data
    batch_size = args.batch_size

    print("[args] gpu=%d" % gpu)
    print("[args] model_name=%s" % model_name)
    print("[args] root_path=%s" % root_path)
    print("[args] json_path=%s" % json_path)
    print("[args] save_path=%s" % save_path)
    print("[args] data=%s" % data)
    print("[args] batch_size=%s" % batch_size)

    device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")

    # Model preparation
    if model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        dim = model.fc.in_features
        model.fc = Net()
    else:
        raise ValueError

    model.to(device)
    model.eval()

    # Coco preparation
    coco = COCO(os.path.join(json_path))
    ids = list(coco.anns.keys())
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img2vec = {}

    with torch.no_grad():
        for i in tqdm(range(0, len(ids), batch_size)):
            img_ids = [coco.anns[ann_id]['image_id'] for ann_id in ids[i:i+batch_size]]
            pathes = [coco.loadImgs(img_id)[0]['file_name'] for img_id in img_ids]
            images = [Image.open(os.path.join(root_path, path)).convert('RGB') for path in pathes]
            images = torch.stack(tuple([transform(image) for image in images]), 0).to(device)

            feats = model(images)
            feats = feats.to(torch.device("cpu"))

            for img_id, feat in zip(img_ids, feats):
                img2vec[img_id] = feat

    torch.save(img2vec, os.path.join(save_path, data + '.' + model_name + '.' + str(dim) + 'd.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    args = parser.parse_args()
    main(args)
