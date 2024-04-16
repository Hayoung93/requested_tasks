import os
import argparse
from tqdm import tqdm
from pprint import pprint

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, default_collate

import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as ttv2
from torchvision import models
from torchmetrics.classification import BinaryConfusionMatrix

import detr_transforms as DT
from COCO_multi_cls import COCOMultiLabelCls
from custom_losses import DiceBCELoss

COCO_CAT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def collate_fn(_data):
     # get largest H and W
    sizes_hw = [d[0].shape[1:] for d in _data]
    hs, ws = list(zip(*sizes_hw))
    H, W = max(hs), max(ws)
    # # zero-pad to match maximum H and W. Pad only right and bottom side to avoid re-computing bbox coordinate
    # psizes = [tuple(H - h for h in hs), tuple(W - w for w in ws)]
    # padded_images = [F.pad(d[0], [0, ps[1], 0, ps[0]]) for d, ps in zip(_data, zip(*psizes))]
    # # images
    # images = torch.stack(padded_images, dim=0)
    images = torch.zeros(len(_data), 3, H, W)
    for i, _d in enumerate(_data):
        h, w = _d[0].shape[1:]
        images[i][:, :h, :w] = _d[0]
    # gts
    labels = torch.zeros(len(_data), 81)
    for i, d in enumerate(_data):
        converted_cat_ids = [COCO_CAT_IDS.index(dd) + 1 for dd in d[1]]
        labels[i][converted_cat_ids] = 1
    # misc - file path, etc.
    others = [d[2] for d in _data]
    return images, labels, others
    # return default_collate([_d[0] for _d in _data]), labels, others

def collate_fn_val(_data):
    sizes_hw = [d[0][0].shape[1:] for d in _data]
    hs, ws = list(zip(*sizes_hw))
    H, W = max(hs), max(ws)
    images = torch.zeros(len(_data), 3, H, W)
    for i, _d in enumerate(_data):
        h, w = _d[0][0].shape[1:]
        images[i][:, :h, :w] = _d[0][0]
    # gts
    labels = torch.zeros(len(_data), 81)
    for i, d in enumerate(_data):
        converted_cat_ids = [COCO_CAT_IDS.index(dd) + 1 for dd in d[1]]
        labels[i][converted_cat_ids] = 1
    # misc - file path, etc.
    others = [d[2] for d in _data]
    return images, labels, others

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--year", type=str, help="COCO year", choices=["2017"])
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
    parser.add_argument("--pretrained", type=str)
    args = parser.parse_args()
    return args


default_settings = {
    "root": "/data/data/MSCoco",
    "year": "2017",
    "batch_size": 4,
    'lr': 2e-4,
    "num_workers": 4,
    'epochs': 50,
    "out_dir": "./checkpoints/",
    "log_name": "debug",
    "save_interval": 5,
    "pretrained": "fasterrcnn_v2"
}


def set_model_defaults(args):
    runtime_args = vars(args)
    for k, v in runtime_args.items():
        if v is None and k in default_settings:
            setattr(args, k, default_settings[k])
    return args


def main(args):
    train_transforms = ttv2.Compose([
        ttv2.RandomResize(480, 800),
        # ttv2.Resize(200),
        # ttv2.Resize(800),
        # ttv2.Resize([800, 800]),
        ttv2.RandomHorizontalFlip(),
        ttv2.ToTensor(),
        ttv2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # val_transforms = ttv2.Compose([ttv2.Resize(800), ttv2.ToTensor(), ttv2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    val_transforms = DT.Compose([DT.RandomResize([800], max_size=1333), DT.ToTensor(), DT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    if args.year == "2017":
        trainset = COCOMultiLabelCls(args.root, "train", train_transforms, args.year)
        valset = COCOMultiLabelCls(args.root, "val", val_transforms, args.year)
    else:
        raise Exception("Not supported VOC year")
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_val)
    model = models.resnet50(num_classes=81)
    model.fc = torch.nn.Sequential(torch.nn.Dropout(0.5), model.fc, torch.nn.Sigmoid())
    if (args.pretrained is not None) and (args.pretrained == "fasterrcnn_v2"):
        cp = models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.get_state_dict()
        cp_backbone = {}
        for k, v in cp.items():
            if k.startswith("backbone.body."):
                cp_backbone[k.replace("backbone.body.", "")] = v
        msg = model.load_state_dict(cp_backbone, strict=False)
        print(msg)
    if args.dataparallel:
        assert torch.cuda.is_available(), "CUDA not available"
        model = torch.nn.DataParallel(model)
    optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=args.lr*0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # criterion = torch.nn.BCELoss()
    criterion = DiceBCELoss()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu_only else "cpu")
    save_dir = os.path.join(args.out_dir, args.log_name)
    os.makedirs(save_dir, exist_ok=True)
    start_epoch = 0
    best_prec = 0
    if args.resume is not None and os.path.isfile(args.resume):
        cp = torch.load(args.resume)
        model.load_state_dict(cp["model"])
        optimizer = optimizer.load_state_dict(cp["optimizer"])
        scheduler = scheduler.load_state_dict(cp["scheduler"])
        start_epoch = cp["epoch"] + 1
        if "best_prec" in cp:
            best_prec = cp["best_prec"]

    if args.eval:
        print("Start evaluation...")
        model = model.to(device)
        model = model.eval()
        metric = BinaryConfusionMatrix().to(device)
        pbar_eval = tqdm(valloader, position=2)
        with torch.inference_mode():
            tns, fps, fns, tps = 0, 0, 0, 0
            for inputs, targets, file_fps in pbar_eval:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                cm = metric((logits > 0.5).to(torch.uint8), targets)
                tn, fp, fn, tp = cm.flatten()
                tns += tn
                fps += fp
                fns += fn
                tps += tp
            precision = tps / (tps + fps)
            recall = tps / (tps + fns)
            print("Precision: {:.6f} | Recall: {:.6f}".format(precision, recall))
        exit(0)

    print("Start training...")
    model = model.to(device)
    pbar_ep = tqdm(range(start_epoch, args.epochs, 1), position=0)
    for epoch in pbar_ep:
        # train
        model = model.train()
        pbar_train = tqdm(trainloader, position=1)
        for inputs, targets, file_fps in pbar_train:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar_train.set_description("[TRAIN] Epoch: {:03d} | Total loss: {:.4f} | LR: {:.9f}".format(epoch, loss.item(), optimizer.param_groups[0]["lr"]))
        scheduler.step()
        # eval
        model = model.eval()
        metric = BinaryConfusionMatrix().to(device)
        pbar_eval = tqdm(valloader, position=2)
        with torch.inference_mode():
            tns, fps, fns, tps = 0, 0, 0, 0
            for inputs, targets, file_fps in pbar_eval:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                cm = metric((logits > 0.5).to(torch.uint8), targets)
                tn, fp, fn, tp = cm.flatten()
                tns += tn
                fps += fp
                fns += fn
                tps += tp
            precision = tps / (tps + fps)
            recall = tps / (tps + fns)
            print("Precision: {:.6f} | Recall: {:.6f}".format(precision, recall))
        # save
        if (epoch + 1) % args.save_interval == 0:
            # avoid saving dataparallel object
            if isinstance(model, torch.nn.DataParallel):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            torch.save({
                "model": model_state_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_prec": best_prec
            }, os.path.join(save_dir, "checkpoint{:03d}.pth".format(epoch)))
        if precision > best_prec:
            best_prec = precision
            # avoid saving dataparallel object
            if isinstance(model, torch.nn.DataParallel):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            torch.save({
                "model": model_state_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_prec": best_prec
            }, os.path.join(save_dir, "checkpoint_best.pth"))
        print("Best precision: {:.6f}".format(best_prec))


if __name__ == "__main__":
    args = get_args()
    args = set_model_defaults(args)
    main(args)
