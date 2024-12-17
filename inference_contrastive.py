import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from torchvision import transforms
from torchvision import models as tvmodels
from torch.utils.data import DataLoader

from config import get_cfg
from dataset import LSPD_Dataset
from models import get_models, ContrastiveModel


@torch.no_grad()
def inference(args, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    testset = LSPD_Dataset(args, cfg, "test", transform)
    testloader = DataLoader(testset, 1, False, num_workers=cfg.run.num_workers)
    swin = tvmodels.swin_t(weights=tvmodels.Swin_T_Weights.IMAGENET1K_V1)
    criterion = torch.nn.CrossEntropyLoss()
    model = ContrastiveModel(cfg, swin, criterion, None)
    model.to(device)
    if (cfg.io.resume is not None) and (cfg.io.resume != "") and (os.path.isfile(cfg.io.resume)):
        cp = torch.load(cfg.io.resume)
        model.model.load_state_dict(cp["model"])
        print("Loaded checkpoint from: {}".format(cfg.io.resume))
    model.eval()

    pbar_iter_test = tqdm(testloader)
    preds = []
    labels = []
    for it, data in enumerate(pbar_iter_test):
        img, label, fp = data
        inputs = img.to(device)
        label = label.to(device)
        outputs, losses = model(inputs, label)
        preds.append(outputs.argmax(dim=1).item())
        labels.append(label.item())
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    acc = (preds == labels).sum() / len(preds)
    f1_score = metrics.f1_score(labels, preds, average=None)
    print("Acc: {:.4f} | Avg. F1 score: {:.4f}".format(acc * 100, f1_score.mean() * 100))
    print("Classwise F1 score: {}".format({k: round(v, 4) for k, v in zip(testset.classes.keys(), f1_score.tolist())}))


if __name__ == "__main__":
    args, cfg = get_cfg()
    inference(args, cfg)
