import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as ttf


class DF40sub(Dataset):
    def __init__(self, args, cfg, mode, transforms):
        self.args = args
        self.cfg = cfg
        self.mode = mode
        assert mode == "test", "Only supports test mode"
        self.transforms = transforms

        fps_fake = [os.path.join(cfg.data.root, "df40-sub", "test", "fake", f) for f in os.listdir(os.path.join(cfg.data.root, "df40-sub", "test", "fake"))]
        fps_real = [os.path.join(cfg.data.root, "df40-sub", "test", "real", f) for f in os.listdir(os.path.join(cfg.data.root, "df40-sub", "test", "real"))]
        self.fps = fps_fake + fps_real
    
    def __len__(self):
        return len(self.fps)
    
    def __getitem__(self, idx):
        fp = self.fps[idx]
        label = 0 if "real" in fp else 1
        img = Image.open(fp).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = ttf.to_tensor(img)
        return img, label, fp

    def collate_fn(self, batch):
        return torch.utils.data.default_collate(batch)
