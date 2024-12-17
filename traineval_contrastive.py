import os
import torch
from tqdm import tqdm
from functools import reduce
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models as tvmodels

from config import get_cfg
from models import ContrastiveModel
from contrastive_loss import ContrastiveLoss
from dataset import LSPD_Dataset, DiverseBatchSampler


def main(args, cfg):
    # train variables
    start_epoch = 0
    best_train_loss = torch.inf
    best_score = 0
    writer = SummaryWriter(os.path.join(cfg.io.save_dir, cfg.io.exp_name))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(30, (0.3, 0.3), (0.7, 1.3), 10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    trainset = LSPD_Dataset(args, cfg, "train", transform)
    valset = LSPD_Dataset(args, cfg, "val", transform_val)
    # train_sampler = DiverseBatchSampler(trainset, cfg.run.batch_size, cfg.data.num_classes)
    # trainloader = DataLoader(trainset, batch_sampler=train_sampler, num_workers=cfg.run.num_workers)
    trainloader = DataLoader(trainset, batch_size=cfg.run.batch_size, shuffle=True, num_workers=cfg.run.num_workers)
    valloader = DataLoader(valset, cfg.run.batch_size, False, num_workers=cfg.run.num_workers)
    # model
    swin = tvmodels.swin_t(weights=tvmodels.Swin_T_Weights.IMAGENET1K_V1)
    if cfg.run.criterion == "ce":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise Exception("Not supported loss: {}".format(cfg.run.criterion))
    criterion_cont = ContrastiveLoss(cfg.data.num_classes, margin=cfg.run.contrastive, pull_positive=True)
    criterions = [criterion, criterion_cont]
    model = ContrastiveModel(cfg, swin, criterion, criterion_cont)
    model.to(device)
    # optimization
    try:
        assert isinstance(cfg.run.lr, float), "Invalid learning rate: {}".format(cfg.run.lr)
        assert isinstance(cfg.run.weight_decay, (float, int)), "Invalid weight decay: {}".format(cfg.run.weight_decay)
        optimizer = eval("torch.optim.{}".format(cfg.run.optimizer))(model.parameters(), lr=cfg.run.lr, weight_decay=cfg.run.weight_decay)
    except AttributeError as e:
        print("Invalid optimizer: {}".format(cfg.run.optimizer))
        raise e
    try:
        assert isinstance(cfg.run.epochs, int), "Invalid epochs: {}".format(cfg.run.epochs)
        scheduler = eval("torch.optim.lr_scheduler.{}".format(cfg.run.scheduler))(optimizer, cfg.run.epochs)
    except AttributeError as e:
        print("Invalid scheduler: {}".format(cfg.run.scheduler))
        raise e
    # pretrain
    if (cfg.io.pretrain is not None) and (cfg.io.pretrain != "") and (os.path.isfile(cfg.io.pretrain)):
        cp = torch.load(cfg.io.pretrain)
        model.model.load_state_dict(cp, strict=False)
        for n, p in model.model.named_parameters():
            p.requires_grad = False
    # resume
    if (cfg.io.resume is not None) and (cfg.io.resume != "") and (os.path.isfile(cfg.io.resume)):
        cp = torch.load(cfg.io.resume)
        model.model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        for ci, criterion in enumerate(criterions):
            criterion.load_state_dict(cp["criterions"][ci])
        scheduler.load_state_dict(cp["scheduler"])
        start_epoch = cp["epoch"] + 1
        best_train_loss = cp["best_train_loss"]
        best_score = cp["best_score"]
    # parallel
    if cfg.run.parallel == "DP":
        model = torch.nn.DataParallel(model)
        parallel = True
    elif cfg.run.parallel == "DDP":
        parallel = True
        raise NotImplementedError
    # loop
    max_it_per_epoch = len(trainloader)
    pbar_epoch = tqdm(range(start_epoch, cfg.run.epochs), position=0)
    for ep in pbar_epoch:
        # train
        model.train()
        pbar_epoch.set_description("Epoch: {}".format(ep))
        ep_loss = 0.0
        pbar_iter_train = tqdm(trainloader, position=1)
        for it, data in enumerate(pbar_iter_train):
            if it > max_it_per_epoch:
                break
            img, label, fp = data
            inputs = img.to(device)
            label = label.to(device)
            outputs, losses = model(inputs, label)
            loss = sum([l.mean() for l in losses.values()])
            running_loss = loss.item()
            ep_loss += running_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar_iter_train.set_description("Iter: {} | Loss: {:.4f}".format(it, running_loss))
        if ep_loss < best_train_loss:
            best_train_loss = ep_loss
        writer.add_scalar("Loss", ep_loss, ep)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], ep)
        pbar_epoch.set_description("Epoch: {} | Total Loss: {} | LR: {}".format(ep, ep_loss, optimizer.param_groups[0]["lr"]))
        scheduler.step()
        # val
        if (ep + 1) % cfg.run.val_interval == 0:
            model.eval()
            pbar_iter_val = tqdm(valloader, position=2)
            with torch.inference_mode():
                classwise_correct = [0] * args.num_classes
                classwise_count = [0] * args.num_classes
                for it, data in enumerate(pbar_iter_val):
                    img, label, fp = data
                    inputs = img.to(device)
                    label = label.to(device)
                    outputs, losses = model(inputs, label)
                    # prob = outputs.softmax(dim=1)
                    pred = outputs.argmax(dim=1)
                    for p, l in zip(pred, label):
                        classwise_correct[l] += (p == l).item()
                        classwise_count[l] += 1
                classwise_acc = torch.tensor(classwise_correct) / torch.tensor(classwise_count)
                for ci in range(1, args.num_classes + 1):
                    writer.add_scalar("Metric/ACC_class-{}".format(ci), classwise_acc[ci - 1].item(), ep)
                if classwise_acc.mean() > best_score:
                    best_score = classwise_acc.mean()
                    torch.save({
                        "model": model.module.model.state_dict() if parallel else model.model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "criterions": [criterion.state_dict() for criterion in criterions],
                        "scheduler": scheduler.state_dict(),
                        "epoch": ep,
                        "best_train_loss": best_train_loss,
                        "best_score": best_score
                    }, os.path.join(cfg.io.save_dir, cfg.io.exp_name, "checkpoint_best.pth"))
            torch.save({
            "model": model.module.model.state_dict() if parallel else model.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "criterions": [criterion.state_dict() for criterion in criterions],
            "scheduler": scheduler.state_dict(),
            "epoch": ep,
            "best_train_loss": best_train_loss,
            "best_score": best_score
        }, os.path.join(cfg.io.save_dir, cfg.io.exp_name, "checkpoint.pth"))
    torch.save({
        "model": model.module.model.state_dict() if parallel else model.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "criterions": [criterion.state_dict() for criterion in criterions],
        "scheduler": scheduler.state_dict(),
        "epoch": ep,
        "best_train_loss": best_train_loss,
        "best_score": best_score
    }, os.path.join(cfg.io.save_dir, cfg.io.exp_name, "checkpoint_last.pth"))


if __name__ == "__main__":
    args, cfg = get_cfg()
    main(args, cfg)
