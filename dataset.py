import os
import json
import torch
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from functools import reduce
from collections import defaultdict
from torch.utils.data import Dataset, Sampler
import torchvision.transforms.functional as ttf


class FaceForensicspp(Dataset):
    def __init__(self, args, cfg, mode, transforms, **kwargs):
        self.args = args
        self.cfg = cfg
        self.mode = mode
        assert mode in ["train", "val", "test", "test_video"], "Not supported mode: {}".format(mode)
        self.transforms = transforms
        self.kwargs = kwargs

        # read train, val, test index split
        with open(os.path.join(cfg.data.root, "{}.json".format(mode.replace("_video", ""))), "r") as f:
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
        if mode == "test_video":
            real_fps_vid = defaultdict(list)
            for fp in real_fps:
                if fp.split("/")[-5] == "FaceShifter": continue  # discard FaceShifter
                vid_idx = fp.split("/")[-2]
                if vid_idx in self.vid_index_flatten:
                    real_fps_vid[vid_idx].append(os.path.join(cfg.data.root, fp))
            self.real_fps = list(dict(sorted(real_fps_vid.items(), key=lambda x: x[0])).values())
            fake_fps_vid = defaultdict(list)
            for fp in fake_fps:
                if fp.split("/")[-5] == "FaceShifter": continue  # discard FaceShifter
                vid_idx = fp.split("/")[-2].split("_")[0]
                if vid_idx in self.vid_index_flatten:
                    fake_fps_vid[vid_idx].append(os.path.join(cfg.data.root, fp))
            self.fake_fps = list(dict(sorted(fake_fps_vid.items(), key=lambda x: x[0])).values())
        else:
            self.real_fps = defaultdict(list)
            for fp in real_fps:
                if fp.split("/")[-5] == "FaceShifter": continue  # discard FaceShifter
                vid_idx = fp.split("/")[-2].split("_")[0]
                if vid_idx in self.vid_index_flatten:
                    self.real_fps[vid_idx].append(os.path.join(cfg.data.root, fp))
            self.fake_fps = defaultdict(list)
            for fp in fake_fps:
                if fp.split("/")[-5] == "FaceShifter": continue  # discard FaceShifter
                vid_idx = fp.split("/")[-2].split("_")[0]
                if vid_idx in self.vid_index_flatten:
                    self.fake_fps[fp.split("/")[-5] + " " + vid_idx].append(os.path.join(cfg.data.root, fp))
            # maximum frame count
            if cfg.data.max_frame_count > 0:
                for k, v in self.real_fps.items():
                    if len(v) > cfg.data.max_frame_count:
                        random.shuffle(v)
                        self.real_fps[k] = v[:cfg.data.max_frame_count]
                for k, v in self.fake_fps.items():
                    if len(v) > cfg.data.max_frame_count:
                        random.shuffle(v)
                        self.fake_fps[k] = v[:cfg.data.max_frame_count]
            # final file paths
            self.real_fps = reduce(lambda x, y: x + y, self.real_fps.values(), [])
            self.fake_fps = reduce(lambda x, y: x + y, self.fake_fps.values(), [])
            # self.real_fps = [os.path.join(cfg.data.root, fp) for fp in real_fps if (fp.split("/")[-2] in self.vid_index_flatten) and (fp.split("/")[-5] != "FaceShifter")]
            # self.fake_fps = [os.path.join(cfg.data.root, fp) for fp in fake_fps if (fp.split("/")[-2].split("_")[0] in self.vid_index_flatten) and (fp.split("/")[-5] != "FaceShifter")]

        if mode == "train":
            oversampling_ratio = round(len(self.fake_fps) / len(self.real_fps))
            self.paths = self.real_fps * oversampling_ratio + self.fake_fps
            self.real_len = len(self.real_fps) * oversampling_ratio
        else:
            self.paths = self.real_fps + self.fake_fps
            self.real_len = len(self.real_fps)

    def __getitem__(self, idx):
        fp = self.paths[idx]
        if idx < self.real_len:
            label = 0  # real
        else:
            label = 1  # fake
        if self.mode == "test_video":
            images = []
            for _fp in fp:
                image = Image.open(_fp).convert("RGB")
                if self.transforms is not None:
                    image = self.transforms(image)
                else:
                    image = ttf.to_tensor(image)
                images.append(image)
            return images, label, fp
        else:
            image = Image.open(fp).convert("RGB")
            if self.transforms is not None:
                if self.mode == "train":
                    image = self.transforms[label](image)
                else:
                    image = self.transforms(image)
            else:
                image = ttf.to_tensor(image)
            return image, label, fp

    def __len__(self):
        return len(self.paths)


class FaceForensicsppDFSD(Dataset):
    def __init__(self, args, cfg, mode, transforms, **kwargs):
        self.args = args
        self.cfg = cfg
        self.mode = mode
        assert mode in ["train", "val", "test"], "Not supported mode: {}".format(mode)
        self.transforms = transforms
        self.kwargs = kwargs

        root = "/data/mnt_ssd/dfsd_dev/dump/{}".format(mode)
        self.real_fps = []
        self.fake_fps = []
        for _dir in os.listdir(root):
            if not os.path.isdir(os.path.join(root, _dir)):
                continue
            files = sorted(os.listdir(os.path.join(root, _dir)))
            if len(_dir) == 3:
                self.real_fps += [os.path.join(root, _dir, f) for f in files]
            else:
                self.fake_fps += [os.path.join(root, _dir, f) for f in files]

        if mode == "train":
            oversampling_ratio = round(len(self.fake_fps) / len(self.real_fps))
            self.paths = self.real_fps * oversampling_ratio + self.fake_fps
            self.real_len = len(self.real_fps) * oversampling_ratio
        else:
            self.paths = self.real_fps + self.fake_fps
            self.real_len = len(self.real_fps)

    def __getitem__(self, idx):
        fp = self.paths[idx]
        if idx < self.real_len:
            label = 0  # real
        else:
            label = 1  # fake
        if self.mode == "test_video":
            images = []
            for _fp in fp:
                image = Image.open(_fp).convert("RGB")
                if self.transforms is not None:
                    image = self.transforms(image)
                else:
                    image = ttf.to_tensor(image)
                images.append(image)
            return images, label, fp
        else:
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
