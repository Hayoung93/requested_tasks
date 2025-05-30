import os
import json
import torch
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter

from config import get_cfg
from models import get_model
from randaug import RandAugment
from datasets.celebv2 import CelebDFv2
from datasets.dataset_gui import CenterCrop
from datasets.ff import FaceForensicspp, FaceForensicsppReal, FaceForensicsppFake, FaceForensicsppBalance, FaceForensicsppVideo, CurriculumSampler


def main(args, cfg):
    # random seed
    torch.manual_seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)
    torch.cuda.manual_seed(cfg.run.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # save cfg and some codes
    os.makedirs(os.path.join(cfg.io.save_dir, cfg.io.exp_name), exist_ok=True)
    with open(os.path.join(cfg.io.save_dir, cfg.io.exp_name, "config.json"), "w") as f:
        json.dump(cfg, f, indent=4)
    shutil.copy(os.path.abspath(__file__), os.path.join(cfg.io.save_dir, cfg.io.exp_name, "traineval.py"))
    # train variables
    start_epoch = 0
    best_train_loss = torch.inf
    best_score = 0
    writer = SummaryWriter(os.path.join(cfg.io.save_dir, cfg.io.exp_name))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data
    # transform_real = transforms.Compose([
    #     transforms.Resize((args.input_size, args.input_size)),
    #     transforms.RandomHorizontalFlip(),
    #     # RandAugment(4, 15),
    #     transforms.RandAugment(1, 10),
    #     transforms.ToTensor(),
    #     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    # transform_fake = transforms.Compose([
    #     transforms.Resize((args.input_size, args.input_size)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandAugment(2, 10),
    #     # transforms.ElasticTransform(),
    #     # transforms.RandomApply(torch.nn.ModuleList([transforms.ElasticTransform()]), p=0.25),
    #     transforms.ToTensor(),
    #     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    transform_real = transforms.Compose([
        CenterCrop((0.8, 0.7)),
        transforms.Resize((args.input_size, args.input_size), interpolation=transforms.InterpolationMode.NEAREST),
        # transforms.ColorJitter(
        #     brightness=0.2,    # 밝기 변형 범위를 줄임: [0.8, 1.2]
        #     contrast=0.2,      # 대비 변형 범위를 줄임
        #     saturation=0.2,    # 채도 변형 범위를 줄임
        #     hue=(-0.05, 0.05)  # 색조 변형 범위를 줄임
        # ),
        # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(
            p=0.5,              # 적용 확률
            scale=(0.02, 0.1),  # erased 영역의 면적 비율 (최대 10%로 제한)
            ratio=(0.3, 3.3),   # erased 영역의 가로세로 비율 (필요에 따라 조정)
            value=0             # erased 영역을 채울 값
        )
    ])
    transform_fake = transform_real
    transform_val = transforms.Compose([
        CenterCrop((0.8, 0.7)),
        transforms.Resize((args.input_size, args.input_size), transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # trainset = FaceForensicspp(args, cfg, "train", transforms=[transform_real, transform_fake])
    # trainset = FaceForensicsppBalance(args, cfg, "train", transforms=[transform_real, transform_fake])
    trainset_real = FaceForensicsppReal(args, cfg, "train", transform_real)
    trainset_fake = FaceForensicsppFake(args, cfg, "train", transform_fake)
    valset = FaceForensicspp(args, cfg, "val", transform_val)

    testsets = [
        FaceForensicspp(args, cfg, "val", transform_val),
        FaceForensicspp(args, cfg, "test", transform_val),
        FaceForensicsppVideo(args, cfg, "test_video", transform_val),
        # CelebDFv2(args, cfg, "test", transform_val),
    ]
    if args.curriculum:
        # Measure difficulty
        if args.load_difficulty != "" and os.path.isfile(args.load_difficulty):
            if args.load_difficulty.endswith(".npy"):
                difficulties = np.load(args.load_difficulty)
            elif args.load_difficulty.endswith(".json"):
                with open(args.load_difficulty, "r") as f:
                    difficulties = json.load(f)
            print("Loaded difficulties from {}".format(args.load_difficulty))
        elif args.load_difficulty_real != "" and os.path.isfile(args.load_difficulty_real) and args.load_difficulty_fake != "" and os.path.isfile(args.load_difficulty_fake):
            if args.load_difficulty_real.endswith(".npy"):
                difficulties_real = np.load(args.load_difficulty_real)
            elif args.load_difficulty_real.endswith(".json"):
                with open(args.load_difficulty_real, "r") as f:
                    difficulties_real = json.load(f)
            if args.load_difficulty_fake.endswith(".npy"):
                difficulties_fake = np.load(args.load_difficulty_fake)
            elif args.load_difficulty_fake.endswith(".json"):
                with open(args.load_difficulty_fake, "r") as f:
                    difficulties_fake = json.load(f)
        else:
            if args.difficulty_function == "difficulty-1":
                # difficulty-1: measure difficulty by pretrained model
                difficulties_real, difficulties_fake = [], []
                fps_real, fps_fake = [], []
                _cfg = argparse.Namespace(model=argparse.Namespace(version="v2-timm", pretrained=False), data=argparse.Namespace(num_classes=args.num_classes))
                pretrained = get_model(_cfg, None)
                cp = torch.load(args.curriculum_pretrained)
                pretrained.load_state_dict(cp["model"])
                pretrained.eval()
                pretrained.to(device)
                print("Measuring difficulty...")
                with torch.no_grad():
                    for di in tqdm(range(len(trainset_real))):
                        img_real, label_real, fp_real = trainset_real[di]
                        inputs_real = img_real.unsqueeze(0).to(device)
                        logits_real, _ = pretrained(inputs_real, None)
                        prob_real = torch.softmax(logits_real, dim=1)
                        fakeness_real = prob_real[0][1].item()
                        sign_real = -1 if prob_real.argmax(dim=1) == label_real else 1
                        difficulty_real = fakeness_real * sign_real
                        difficulties_real.append(difficulty_real)
                        fps_real.append(fp_real)
                    for di in tqdm(range(len(trainset_fake))):
                        img_fake, label_fake, fp_fake = trainset_fake[di]
                        inputs_fake = img_fake.unsqueeze(0).to(device)
                        logits_fake, _ = pretrained(inputs_fake, None)
                        prob_fake = torch.softmax(logits_fake, dim=1)
                        sign_fake = -1 if prob_fake.argmax(dim=1) == label_fake else 1
                        fakeness_fake = prob_fake[0][1].item()
                        difficulty_fake = fakeness_fake * sign_fake
                        difficulties_fake.append(difficulty_fake)
                        fps_fake.append(fp_fake)
                print("Done.")
            else:
                raise NotImplementedError("Not implemented difficulty function: {}".format(args.difficulty_function))
        sampler_train_real = CurriculumSampler(args, difficulties_real)
        sampler_train_fake = CurriculumSampler(args, difficulties_fake)
        # trainloader = DataLoader(trainset, batch_size=cfg.run.batch_size, num_workers=cfg.run.num_workers, sampler=sampler_train, collate_fn=trainset.collate_fn, worker_init_fn=trainset.worker_init_fn)
        trainloader_real = DataLoader(trainset_real, batch_size=cfg.run.batch_size, num_workers=cfg.run.num_workers, sampler=sampler_train_real, collate_fn=trainset_real.collate_fn, worker_init_fn=trainset_real.worker_init_fn)
        trainloader_fake = DataLoader(trainset_fake, batch_size=cfg.run.batch_size, num_workers=cfg.run.num_workers, sampler=sampler_train_fake, collate_fn=trainset_fake.collate_fn, worker_init_fn=trainset_fake.worker_init_fn)
    else:
        # trainloader = DataLoader(trainset, batch_size=cfg.run.batch_size, shuffle=True, num_workers=cfg.run.num_workers, collate_fn=trainset.collate_fn)
        trainloader_real = DataLoader(trainset_real, batch_size=cfg.run.batch_size, shuffle=True, num_workers=cfg.run.num_workers, collate_fn=trainset_real.collate_fn)
        trainloader_fake = DataLoader(trainset_fake, batch_size=cfg.run.batch_size, shuffle=True, num_workers=cfg.run.num_workers, collate_fn=trainset_fake.collate_fn)
    valloader = DataLoader(valset, cfg.run.batch_size, False, num_workers=cfg.run.num_workers)
    testloaders = [DataLoader(testset, 1, False, collate_fn=testset.collate_fn) for testset in testsets]
    # valloader = DataLoader(valset, cfg.run.batch_size, False, num_workers=cfg.run.num_workers, collate_fn=valset.collate_fn, worker_init_fn=valset.worker_init_fn)
    # testloader = DataLoader(testset, 1, False, num_workers=cfg.run.num_workers, collate_fn=testset.collate_fn)
    print("# trainset: {} | # valset: {} | # testset: {}".format(len(trainset_real) + len(trainset_fake), len(valset), [len(testset) for testset in testsets]))

    # model
    if cfg.run.criterion == "ce":
        criterion = torch.nn.CrossEntropyLoss()
    elif cfg.run.criterion == "ce-weight":
        criterion = torch.nn.CrossEntropyLoss(torch.tensor([5.0, 1.0], device=device))
    else:
        raise Exception("Not supported loss: {}".format(cfg.run.criterion))
    criterions = [criterion,]
    model = get_model(cfg, criterions)
    model.to(device)
    # optimization
    assert isinstance(cfg.run.lr, float), "Invalid learning rate: {}".format(cfg.run.lr)
    assert isinstance(cfg.run.weight_decay, (float, int)), "Invalid weight decay: {}".format(cfg.run.weight_decay)
    torch_optimizers = dir(torch.optim)
    torch_optimizers = [attr for attr in torch_optimizers if not attr.startswith('__') and callable(getattr(torch.optim, attr))]
    if cfg.run.optimizer in torch_optimizers:
        optimizer = eval("torch.optim.{}".format(cfg.run.optimizer))(model.parameters(), lr=cfg.run.lr, weight_decay=cfg.run.weight_decay)
    elif cfg.run.optimizer == "SAM":
        from src.utils.sam import SAM
        base_optimizer = torch.optim.Adam
        optimizer = SAM(model.parameters(), base_optimizer, lr=cfg.run.lr, weight_decay=cfg.run.weight_decay)
    else:
        raise Exception("Invalid optimizer: {}".format(cfg.run.optimizer))
    try:
        assert isinstance(cfg.run.epochs, int), "Invalid epochs: {}".format(cfg.run.epochs)
        scheduler = eval("torch.optim.lr_scheduler.{}".format(cfg.run.scheduler))(optimizer, cfg.run.epochs)
    except AttributeError as e:
        try:
            scheduler = eval(cfg.run.scheduler + "(optimizer, cfg.run.epochs, round(cfg.run.epochs * 0.75))")
        except Exception as e2:
            print("Invalid scheduler: {}, disabled LR scheduling.".format(cfg.run.scheduler))
            scheduler = None
    # resume
    if (cfg.io.resume is not None) and (cfg.io.resume != "") and (os.path.isfile(cfg.io.resume)):
        cp = torch.load(cfg.io.resume)
        if "model" in cp:
            model.load_state_dict(cp["model"])
        else:
            model.load_state_dict(cp)
        if "optimizer" in cp:
            optimizer.load_state_dict(cp["optimizer"])
        for ci, criterion in enumerate(criterions):
            if "criterions" in cp:
                criterion.load_state_dict(cp["criterions"][ci])
        if "scheduler" in cp and scheduler is not None:
            scheduler.load_state_dict(cp["scheduler"])
        if "epoch" in cp:
            start_epoch = cp["epoch"] + 1
        if "best_train_loss" in cp:
            best_train_loss = cp["best_train_loss"]
        if "best_score" in cp:
            best_score = cp["best_score"]
    # parallel
    if cfg.run.parallel == "DP":
        model = torch.nn.DataParallel(model)
        parallel = True
    elif cfg.run.parallel == "DDP":
        parallel = True
        raise NotImplementedError
    # loop
    if args.curriculum:
        sampler_train_real.sample_data()
        sampler_train_fake.sample_data()
    max_it_per_epoch = len(trainloader_real)
    pbar_epoch = tqdm(range(start_epoch, cfg.run.epochs), position=0)
    for ep in pbar_epoch:
        # train
        model.train()
        pbar_epoch.set_description("Epoch: {}".format(ep))
        ep_loss = 0.0
        classwise_correct = [0] * args.num_classes
        classwise_count = [0] * args.num_classes
        pbar_iter_train = tqdm(trainloader_real, position=1)
        for it, (data_real, data_fake) in enumerate(zip(pbar_iter_train, trainloader_fake)):
            if it > max_it_per_epoch:
                break
            img_real, label_real, fp_real = data_real
            img_fake, label_fake, fp_fake = data_fake
            inputs = torch.cat([img_real.to(device), img_fake.to(device)], dim=0)
            label = torch.cat([label_real.to(device), label_fake.to(device)], dim=0)
            outputs, losses = model(inputs, label)
            pred = outputs.argmax(dim=1)
            for p, l in zip(pred, label):
                classwise_correct[l] += (p == l).item()
                classwise_count[l] += 1
            loss = sum([l.mean() for l in losses.values()])
            running_loss = loss.item()
            ep_loss += running_loss
            optimizer.zero_grad()
            loss.backward()
            if cfg.run.optimizer == "SAM":
                optimizer.first_step(zero_grad=True)
                outputs, losses = model(inputs, label)
                loss = sum([l.mean() for l in losses.values()])
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()
            pbar_iter_train.set_description("Iter: {} | Loss: {:.4f}".format(it, running_loss))
        classwise_acc = torch.tensor(classwise_correct) / torch.tensor(classwise_count)
        if ep_loss < best_train_loss:
            best_train_loss = ep_loss
        writer.add_scalar("Train/Loss", ep_loss, ep)
        writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], ep)
        for ci in range(1, args.num_classes + 1):
            writer.add_scalar("Train/ACC_class-{}".format(ci), classwise_acc[ci - 1].item(), ep)
        pbar_epoch.set_description("Epoch: {} | Total Loss: {} | LR: {}".format(ep, ep_loss, optimizer.param_groups[0]["lr"]))
        if scheduler is not None:
            scheduler.step()
        # val
        if (ep + 1) % cfg.run.val_interval == 0:
            model.eval()
            pbar_iter_val = tqdm(valloader, position=2)
            with torch.inference_mode():
                classwise_correct = [0] * args.num_classes
                classwise_count = [0] * args.num_classes
                labels = []
                preds = []
                for it, data in enumerate(pbar_iter_val):
                    img, label, fp = data
                    # img, label = data["img"], data["label"]
                    inputs = img.to(device)
                    labels.extend(label.tolist())
                    label = label.to(device)
                    outputs, losses = model(inputs, label)
                    pred = outputs.argmax(dim=1)
                    preds.extend(outputs.softmax(1)[:, 1].cpu().data.numpy().tolist())
                    for p, l in zip(pred, label):
                        classwise_correct[l] += (p == l).item()
                        classwise_count[l] += 1
                classwise_acc = torch.tensor(classwise_correct) / torch.tensor(classwise_count)
                auc = roc_auc_score(labels, preds)
                writer.add_scalar("Metric/AUC", auc, ep)
                for ci in range(1, args.num_classes + 1):
                    writer.add_scalar("Metric/ACC_class-{}".format(ci), classwise_acc[ci - 1].item(), ep)
                if classwise_acc.mean() > best_score:
                    best_score = classwise_acc.mean()
                    torch.save({
                        "model": model.module.state_dict() if parallel else model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "criterions": [criterion.state_dict() for criterion in criterions],
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "epoch": ep,
                        "best_train_loss": best_train_loss,
                        "best_score": best_score
                    }, os.path.join(cfg.io.save_dir, cfg.io.exp_name, "checkpoint_best.pth"))
            torch.save({
            "model": model.module.state_dict() if parallel else model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "criterions": [criterion.state_dict() for criterion in criterions],
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": ep,
            "best_train_loss": best_train_loss,
            "best_score": best_score
        }, os.path.join(cfg.io.save_dir, cfg.io.exp_name, "checkpoint.pth"))
        if args.curriculum:
            sampler_train_real.sample_data()
            sampler_train_fake.sample_data()
            max_it_per_epoch = len(trainloader_real)
    # test
    if start_epoch >= cfg.run.epochs:
        ep = start_epoch
    # save last checkpoint
    torch.save({
        "model": model.module.state_dict() if parallel else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "criterions": [criterion.state_dict() for criterion in criterions],
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": ep,
        "best_train_loss": best_train_loss,
        "best_score": best_score
    }, os.path.join(cfg.io.save_dir, cfg.io.exp_name, "checkpoint_last.pth"))
    model.eval()
    for testloader in testloaders:
        pbar_iter_test = tqdm(testloader, position=2)
        with torch.inference_mode():
            classwise_correct = [0] * args.num_classes
            classwise_count = [0] * args.num_classes
            labels = []
            preds = []
            for it, data in enumerate(pbar_iter_test):
                img, label, fp = data
                # img, label = data["img"], data["label"]
                inputs = img.to(device)
                labels.extend(label.tolist())
                label = label.to(device)
                outputs, losses = model(inputs, label)
                pred = outputs.argmax(dim=1)
                preds.extend(outputs.softmax(1)[:, 1].cpu().data.numpy().tolist())
                for p, l in zip(pred, label):
                    classwise_correct[l] += (p == l).item()
                    classwise_count[l] += 1
            classwise_acc = torch.tensor(classwise_correct) / torch.tensor(classwise_count)
            auc = roc_auc_score(labels, preds)
            writer.add_scalar("Test/{}/AUC".format(type(testloader.dataset).__name__), auc, ep)
            for ci in range(1, args.num_classes + 1):
                writer.add_scalar("Test/{}/ACC_class-{}".format(type(testloader.dataset).__name__, ci), classwise_acc[ci - 1].item(), ep)
        print("Dataset: {} | Test AUC: {:.6f}  |  Test real acc: {:.6f}  |  Test fake acc: {:.6f}".format(
            type(testloader.dataset).__name__, auc, classwise_acc[0].item(), classwise_acc[1].item()))


if __name__ == "__main__":
    args, cfg = get_cfg()
    main(args, cfg)
