import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
# from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as ttf


class COCOMultiLabelCls(Dataset):
    def __init__(self, root, mode, transforms, year="2017"):
        self.root = root
        self.mode = mode
        self.transforms = transforms
        self.year = year

        ann_file = os.path.join(root, "annotations", "instances_{}{}.json".format(mode, year))
        with open(ann_file, "r") as f:
            coco_json = json.load(f)
        self.imgs = defaultdict()
        self.anns = defaultdict(list)
        self.ids = set()
        for img in tqdm(coco_json["images"]):
            _id = img["id"]
            self.imgs[_id] = img["file_name"]
        for ann in tqdm(coco_json["annotations"]):
            img_id = ann["image_id"]
            self.anns[img_id].append(ann["category_id"])
            self.ids.add(img_id)
        self.imgs, self.anns = dict(self.imgs), dict(self.anns)
        self.ids = list(self.ids)
        del coco_json
        # self.coco = COCO(ann_file)
        # self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, idx):
        _id = self.ids[idx]
        # img_fn = self.coco.loadImgs(_id)[0]["file_name"]
        img_fn = self.imgs[_id]
        img_fp = os.path.join(self.root, self.mode + self.year, img_fn)
        # ann_ids = self.coco.getAnnIds(_id)
        # anns = self.coco.loadAnns(ann_ids)
        # cat_ids = list(map(lambda x: x["category_id"], anns))
        cat_ids = self.anns[_id]

        img = Image.open(img_fp).convert("RGB")
        # img = np.asarray(img)
        if self.transforms is not None:
            img = self.transforms(img)
            # img = self.transforms(image=img)["image"]
        # img = torch.from_numpy(img).permute(2, 1, 0) / 255.
        # img = ttf.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return img, cat_ids, img_fp

    def __len__(self):
        return len(self.ids)
        # return len(self.coco_info)
