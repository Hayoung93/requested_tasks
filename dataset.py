import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


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
        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = transforms.functional.to_tensor(img)
        classname = fp.split("/")[-2].split("_")[0]
        label = self.classes[classname]
        return img, label, fp

    def __len__(self):
        return len(self.files)