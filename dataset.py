import os
import json
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Sampler
import torchvision.transforms.functional as ttf


class FaceForensicspp(Dataset):
    def __init__(self, args, cfg, mode, transforms, **kwargs):
        self.args = args
        self.cfg = cfg
        self.mode = mode
        assert mode in ["train", "val", "test"], "Not supported mode: {}".format(mode)
        self.transforms = transforms
        self.kwargs = kwargs

        # read train, val, test index split
        with open(os.path.join(cfg.data.root, "{}.json".format(mode)), "r") as f:
            vid_index = json.load(f)
        vid_index_flatten = []
        for (v1, v2) in vid_index:
            vid_index_flatten.append(v1)
            vid_index_flatten.append(v2)
        self.vid_index_flatten = vid_index_flatten

        # read real, fake frame paths
        with open(os.path.join(cfg.data.root, "real_{}_faces.txt".format(cfg.data.quality)), "r") as f:
            real_fps = f.read().splitlines()
        with open(os.path.join(cfg.data.root, "fake_{}_faces.txt".format(cfg.data.quality)), "r") as f:
            fake_fps = f.read().splitlines()

        # filter real and fake paths with splitted index
        self.real_fps = [os.path.join(cfg.data.root, fp) for fp in real_fps if fp.split("/")[-2] in self.vid_index_flatten]
        self.fake_fps = [os.path.join(cfg.data.root, fp) for fp in fake_fps if fp.split("/")[-2].split("_")[0] in self.vid_index_flatten]

        oversampling_ratio = round(len(self.fake_fps) / len(self.real_fps))
        self.real_len = len(self.real_fps) * oversampling_ratio

        self.paths = self.real_fps * oversampling_ratio + self.fake_fps

    def __getitem__(self, idx):
        fp = self.paths[idx]
        if idx < self.real_len:
            label = 0  # real
        else:
            label = 1  # fake
        image = Image.open(fp).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        else:
            image = ttf.to_tensor(image)
        return image, label, fp

    def __len__(self):
        return len(self.paths)


class CurriculumSampler(Sampler):
    def __init__(self, args, difficulties):
        self.args = args
        self.difficulties = difficulties

        self.epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __iter__(self):
        for di in self.dataset_idx:
            yield di

    def __len__(self):
        return len(self.dataset_idx)
    
    def sample_data(self):
        # sample data based on probabilities
        self.pace()
        # self.dataset_idx = np.random.choice(list(range(len(self.difficulties))), len(self.diff_idx_selected), False, self.probabilities)
        np.random.shuffle(self.diff_idx_selected)
        self.dataset_idx = self.diff_idx_selected.copy()
        self.epoch += 1

    def pace(self):
        # set probability of each sample to be selected
        if self.args.pace_function == "pace-1":
            diff = np.asarray(self.difficulties)
            diff_idx = diff.argsort()
            if self.epoch < self.args.milestones[0]:
                diff_idx_selected = diff_idx[:int(len(diff_idx) * 0.3)]
            elif self.epoch < self.args.milestones[1]:
                # diff_idx_selected = diff_idx[int(len(diff_idx) * 0.3):int(len(diff_idx) * 0.7)]
                diff_idx_selected = diff_idx
            else:
                diff_idx_selected = diff_idx[int(len(diff_idx) * 0.7):]
            prob = np.zeros(len(self.difficulties))
            prob[diff_idx_selected] = 1
            prob = prob / len(diff_idx_selected)
            self.probabilities = prob
            self.diff_idx_selected = diff_idx_selected
        else:
            raise NotImplementedError("Not implemented pace function: {}".format(self.args.pace_function))
