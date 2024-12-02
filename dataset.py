import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


def project_depth(img, fp_rel, data_root):
    fp_depth = os.path.join(data_root, "vis_depth", fp_rel.replace(".jpg", ".png"))
    img_d = Image.open(fp_depth).convert("RGB")

    img = np.asarray(img, dtype=np.uint16)
    img_d = np.asarray(img_d, dtype=np.uint16) / 255.
    img_p = img * img_d
    img_p = img_p.astype(np.uint8)
    return Image.fromarray(img_p)


class LSPD_Dataset(Dataset):
    def __init__(self, args, cfg, mode, transforms):
        self.args = args
        self.cfg = cfg
        assert mode in ["train", "val", "test"], "Unknown mode: {}".format(mode)
        self.mode = mode
        self.transforms = transforms

        filelist = eval("self.args.{}_filename".format(mode))
        with open(os.path.join(args.root, filelist), "r") as f:
            files = f.read().splitlines()
        self.files = files

        classes = set()
        for file in files:
            classes.add(file.split("/")[0].split("_")[0])
        assert len(classes) == args.num_classes, "Number of classes doesn't match: {} | {}".format(classes, args.num_classes)
        self.classes = {c: i for i, c in enumerate(sorted(list(classes)))}
    
    def __getitem__(self, idx):
        fp = os.path.join(self.args.root, self.files[idx])
        img = Image.open(fp).convert("RGB")
        # project depth map to enhance foreground
        img = project_depth(img, self.files[idx], self.args.root)
        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = transforms.functional.to_tensor(img)
        classname = fp.split("/")[-2].split("_")[0]
        label = self.classes[classname]
        return img, label, fp

    def __len__(self):
        return len(self.files)