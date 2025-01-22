import os
import json
from PIL import Image
from torch.utils.data import Dataset
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
        with open(os.path.join(cfg.data.root, "real_{}.txt".format(cfg.data.quality)), "r") as f:
            real_fps = f.read().splitlines()
        with open(os.path.join(cfg.data.root, "fake_{}.txt".format(cfg.data.quality)), "r") as f:
            fake_fps = f.read().splitlines()

        # filter real and fake paths with splitted index
        self.real_fps = [os.path.join(cfg.data.root, fp) for fp in real_fps if fp.split("/")[-2] in self.vid_index_flatten]
        self.fake_fps = [os.path.join(cfg.data.root, fp) for fp in fake_fps if fp.split("/")[-2].split("_")[0] in self.vid_index_flatten]

        oversampling_ratio = round(len(self.real_fps) / len(self.fake_fps))

        self.paths = self.real_fps * oversampling_ratio + self.fake_fps

    def __getitem__(self, idx):
        fp = self.paths[idx]
        if idx < len(self.real_fps):
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
