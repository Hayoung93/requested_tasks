import os
import torch
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset
import torchvision.transforms.functional as ttf


class CelebDFv2(Dataset):
    """Load Celeb-DF-v2 dataset including videos under YouTube-real directory
    """
    def __init__(self, args, cfg, mode, transforms, **kwargs):
        self.args = args
        self.cfg = cfg
        self.mode = mode
        assert mode in ["test", "test_mini", "test_video"], "Not supported mode: {}".format(mode)
        self.transforms = transforms
        self.kwargs = kwargs

        with open(os.path.join(cfg.data.root, "Celeb-DF_v2", "testset", "{}_faces_list.txt".format(mode)), "r") as f:
            files_faces = f.read().splitlines()
        
        if mode in ["test", "test_mini"]:
            self.files = files_faces
            self.labels = list(map(lambda x: 0 if x.split("/")[0].endswith("-real") else 1, files_faces))
        elif mode == "test_video":
            self.files = defaultdict(list)  # {video_name: [face1, face2, ...]}
            for file in files_faces:
                vid_dirpath = "/".join(file.split("/")[:-1])
                self.files[vid_dirpath].append(file)
            self.files = {k: sorted(self.files[k]) for k in sorted(self.files.keys())}
            self.labels = [0 if k.split("/")[0].endswith("-real") else 1 for k in self.files.keys()]
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.mode in ["test", "test_mini"]:
            file_fps = [os.path.join(self.cfg.data.root, "Celeb-DF_v2", "testset", self.files[idx])]
        elif self.mode == "test_video":
            file_fps = self.files[idx]
        label = self.labels[idx]

        images = []
        for file_fp in file_fps:
            img = Image.open(file_fp).convert("RGB")
            if self.transforms is not None:
                img = self.transforms(img)
            else:
                img = ttf.to_tensor(img)
            images.append(img)
        images = torch.stack(images, dim=0)
        return images, label, file_fps

    def collate_fn(self, batch):
        images, labels, file_fps = zip(*batch)
        images = torch.cat(images, dim=0)
        labels = torch.tensor(labels)
        return images, labels, file_fps
