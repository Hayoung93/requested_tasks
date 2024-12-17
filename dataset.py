import os
import sys
import cv2
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict
from torch.utils.data import Dataset, Sampler


"""def project_depth(img, fp_rel, data_root, input_size):
    fp_depth = os.path.join(data_root, "vis_depth", fp_rel.replace(".jpg", ".png"))
    img_d = Image.open(fp_depth).convert("RGB")

    img = transforms.functional.resize(img, (input_size, input_size))
    img_d = transforms.functional.resize(img_d, (input_size, input_size))

    img = np.asarray(img, dtype=np.uint16)
    img_d = np.asarray(img_d, dtype=np.uint16) / 255.
    h, w = img.shape[:2]
    h_d, w_d = img_d.shape[:2]
    if h != h_d and w == h_d and h == w_d:
        img = img.transpose(1, 0, 2)
    img_p = img * img_d
    img_p = img_p.astype(np.uint8)
    return Image.fromarray(img_p)


def concat_depth(img, fp_rel, data_root, input_size):
    fp_depth = os.path.join(data_root, "vis_depth", fp_rel.replace(".jpg", ".png"))
    img_d = Image.open(fp_depth).convert("L")

    img = transforms.functional.resize(img, (input_size, input_size))
    img_d = transforms.functional.resize(img_d, (input_size, input_size))

    img = np.asarray(img, dtype=np.uint16)
    img_d = np.asarray(img_d, dtype=np.uint16)[..., None] / 255.
    h, w = img.shape[:2]
    h_d, w_d = img_d.shape[:2]
    if h != h_d and w == h_d and h == w_d:
        img = img.transpose(1, 0, 2)
    img_c = np.concatenate([img, img_d], axis=2)
    img_c = img_c.astype(np.uint8)
    return Image.fromarray(img_c)"""


class DiverseBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_classes):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.class_indices = defaultdict(list)
        
        for idx, (_, label, _) in enumerate(tqdm(dataset)):
            self.class_indices[label].append(idx)
        
        self.class_indices = {k: v for k, v in self.class_indices.items() if len(v) > 0}
        self.num_samples_per_class = batch_size // num_classes

    def __iter__(self):
        indices = []
        for _ in range(len(self.dataset) // self.batch_size):
            batch = []
            for class_indices in self.class_indices.values():
                batch.extend(random.sample(class_indices, self.num_samples_per_class))
            random.shuffle(batch)
            indices.extend(batch)
        return iter(indices)

    def __len__(self):
        return len(self.dataset) // self.batch_size


class LSPD_Dataset(Dataset):
    def __init__(self, args, cfg, mode, transforms=None, max_count=None):
        self.args = args
        self.cfg = cfg
        assert mode in ["train", "val", "val-tsne", "test"], "Unknown mode: {}".format(mode)
        self.mode = mode
        self.transforms = transforms

        filelist = eval("self.args.{}_filename".format(mode.replace("-tsne", "")))
        with open(os.path.join(args.root, filelist), "r") as f:
            files = f.read().splitlines()
        self.files = files

        classes = set()
        for file in files:
            classes.add(file.split("/")[0].split("_")[0])
        if (len(classes) != args.num_classes) and (args.num_classes == 2):
            classes = sorted(list(classes))  # ['drawing', 'hentai', 'normal', 'porn', 'sexy']
            self.classes = {classes[0]: 0, classes[2]: 0, classes[1]: 1, classes[3]: 1, classes[4]: 1}
        else:
            assert len(classes) == args.num_classes, "Number of classes doesn't match: {} | {}".format(classes, args.num_classes)
            self.classes = {c: i for i, c in enumerate(sorted(list(classes)))}

        if mode == "val-tsne":
            classwise_files = {k: [] for k in self.classes.keys()}
            for file in self.files:
                classname = file.split("/")[0].split("_")[0]
                if len(classwise_files[classname]) < max_count:
                    classwise_files[classname].append(file)
            self.files = []
            for _, v in classwise_files.items():
                self.files.extend(v)
    
    def __getitem__(self, idx):
        fp = os.path.join(self.args.root, self.files[idx])
        img = Image.open(fp).convert("RGB")
        # if self.mode == "train":
        #     # project depth map to enhance foreground
        #     if random.random() > 0.5:
        #         img = project_depth(img, self.files[idx], self.args.root, self.args.input_size)
        # # concatenate depth map
        # img = concat_depth(img, self.files[idx], self.args.root, self.args.input_size)
        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = transforms.functional.to_tensor(img)
        classname = fp.split("/")[-2].split("_")[0]
        label = self.classes[classname]
        return img, label, fp

    def __len__(self):
        return len(self.files)