import os
import argparse
from tqdm import tqdm

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as ttv2
from torchvision import models

from VOC_det_11cls import VOCDet11cls


class RandomSelect(torch.nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, *inputs):
        img, target = inputs[0]  # Warning! Have no idea if calling [0] will work for all cases
        selected_idx = torch.randint(0, len(self.transforms), (1, ))
        return self.transforms[selected_idx](img, target)


def collate_fn(_data):
    # get largest H and W
    sizes_hw = [d[1]["boxes"].spatial_size for d in _data]
    hs, ws = list(zip(*sizes_hw))
    H, W = max(hs), max(ws)
    # zero-pad to match maximum H and W. Pad only right and bottom side to avoid re-computing bbox coordinate
    psizes = [tuple(H - h for h in hs), tuple(W - w for w in ws)]
    padded_images = [F.pad(d[0], [0, ps[1], 0, ps[0]]) for d, ps in zip(_data, zip(*psizes))]
    # images
    images = torch.stack(padded_images, dim=0)
    # gts - spatial size must be modified to maximum H and W, if spatial size is used somewhere
    gts = [{"boxes": d[1]["boxes"].data, "labels": torch.as_tensor(d[1]["labels"])} for d in _data]
    # misc - file path, etc.
    others = [d[2] for d in _data]
    return images, gts, others


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/data/data/VOC_det")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--log_name", type=str)
    parser.add_argument("--save_interval", type=int)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--use_cpu_only", action="store_true")
    parser.add_argument("--dataparallel", action="store_true")
    args = parser.parse_args()
    return args


default_settings = {
    "batch_size": 4,
    'lr': 2e-4,
    "num_workers": 4,
    'epochs': 50,
    "out_dir": "./checkpoints/",
    "log_name": "debug",
    "save_interval": 5
}


def set_model_defaults(args):
    runtime_args = vars(args)
    for k, v in runtime_args.items():
        if v is None and k in default_settings:
            setattr(args, k, default_settings[k])
    return args


def main(args):
    train_transforms = ttv2.Compose([
        ttv2.RandomHorizontalFlip(),
        RandomSelect([
            ttv2.RandomResize(480, 800),
            ttv2.Compose([
                RandomSelect([
                    ttv2.Compose([ttv2.Resize(400), ttv2.RandomResizedCrop(400, scale=(384 / 400, 1.0))]),
                    ttv2.Compose([ttv2.Resize(500), ttv2.RandomResizedCrop(500, scale=(384 / 500, 1.0))]),
                    ttv2.Compose([ttv2.Resize(600), ttv2.RandomResizedCrop(600, scale=(384 / 600, 1.0))])
                ]),
                ttv2.RandomResize(480, 800)
            ])]),
        ttv2.ToTensor(),
        ttv2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = ttv2.Compose([ttv2.Resize(800), ttv2.ToTensor(), ttv2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    trainset = VOCDet11cls(args.root, "train", train_transforms)
    valset = VOCDet11cls(args.root, "val", val_transforms)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    valloader = DataLoader(valset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    fpn_weights_80 = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT.get_state_dict(progress=True)
    fpn_weights_80 = {k: v for k, v in fpn_weights_80.items() if not ("roi_heads.box_predictor" in k)}
    model = models.detection.fasterrcnn_resnet50_fpn(weights=None,
                                                    weights_backbone=models.ResNet50_Weights.IMAGENET1K_V1,
                                                    trainable_backbone_layer=None,
                                                    num_classes=12)
    msg = model.load_state_dict(fpn_weights_80, strict=False)
    print(msg)
    if args.dataparallel:
        assert torch.cuda.is_available(), "CUDA not available"
        model = torch.nn.DataParallel(model)
    optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=args.lr*0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu_only else "cpu")
    start_epoch = 0
    best_ap = 0
    if args.resume is not None and os.path.isfile(args.resume):
        cp = torch.load(args.resume)
        model = model.load_state_dict(cp["model"])
        optimizer = optimizer.load_state_dict(cp["optimizer"])
        scheduler = scheduler.load_state_dict(cp["scheduler"])
        start_epoch = cp["epoch"] + 1
        best_ap = cp["best_iou"]
    
    if args.eval:
        pass

    print("Start training...")
    model = model.to(device)
    pbar_ep = tqdm(range(start_epoch, args.epochs, 1), position=0)
    for epoch in pbar_ep:
        # train
        model = model.train()
        pbar_train = tqdm(trainloader, position=1)
        for inputs, targets, file_fps in pbar_train:
            inputs = inputs.to(device)
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            loss_dict = model(inputs, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            pbar_ep.set_description("Epoch: {:03d} | Total loss: {:.4f} | LR: {:.9f}".format(epoch, losses.item(), optimizer.param_groups[0]["lr"]))
        scheduler.step()
        # eval
        model = model.eval()
        pbar_eval = tqdm(valloader)
        with torch.inference_mode():
            for inputs, targets, file_fps in pbar_eval:
                inputs = inputs.to(device)
                targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
                loss_dict = model(inputs, targets)
        # save
        if (epoch + 1) % args.save_interval == 0:
            # avoid saving dataparallel object
            if isinstance(model, torch.nn.DataParallel):
                pass
            else:
                pass
        if ap > best_ap:
            # avoid saving dataparallel object
            if isinstance(model, torch.nn.DataParallel):
                pass
            else:
                pass


if __name__ == "__main__":
    args = get_args()
    args = set_model_defaults(args)
    main(args)
