import os
import random
import torch
from torch.nn import functional as F
from torchvision import transforms
from torchvision import transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm

from config import get_cfg
from models import get_models
from dataset import LSPD_Dataset


def get_valid_ratio(mask):
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio


@torch.no_grad()
def tsne(args, cfg):
    max_count = 300
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_models(cfg)
    model.to(device)
    if (cfg.io.resume is not None) and (cfg.io.resume != "") and (os.path.isfile(cfg.io.resume)):
        cp = torch.load(cfg.io.resume)
        model.load_state_dict(cp["model"])
        print("Loaded checkpoint from: {}".format(cfg.io.resume))
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    valset = LSPD_Dataset(args, cfg, "val-tsne", transform, max_count)
    valloader = DataLoader(valset, cfg.run.batch_size, False, num_workers=cfg.run.num_workers, drop_last=False)
    idx2classes = {v: k for k, v in valloader.dataset.classes.items()}

    zs = []
    fps = []
    for data in tqdm(valloader):
        img, label, fp = data
        img = img.to(device)
        z = forward_part(model, img)
        zs.append(z.cpu())
        fps.extend(fp)
    X = torch.cat(zs, dim=0)

    print("Plotting..")
    colormaps = plt.get_cmap("Paired", 5)
    group_len = len(X) // 5
    for perp in [100.0]:
        X_emb = TSNE(init="pca", perplexity=perp).fit_transform(X[:group_len * 5].cpu())
        fig, ax = plt.subplots(1)
        for i in range(5):
            ax.scatter(X_emb[i*group_len:(i+1)*group_len, 0], X_emb[i*group_len:(i+1)*group_len, 1], label=idx2classes[i], s=10, facecolors=colormaps(i / 5.0))
            for j in range(i*group_len, (i+1)*group_len):
                ax.annotate("/".join(fps[j].split("/")[-2:]), (X_emb[j, 0], X_emb[j, 1]), fontsize=5, alpha=0.6)
        fig.set_size_inches(100, 100)
        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig("temp-{}-{}.png".format(max_count, int(perp)), dpi=200, bbox_inches='tight')
    print("Done.")


@torch.no_grad()
def forward_part(model, img):
    z = model.features(img).permute(0, 3, 1, 2)
    z = torch.nn.functional.adaptive_avg_pool2d(z, (1, 1)).squeeze(-1).squeeze(-1)
    return z


if __name__ == "__main__":
    args, cfg = get_cfg()
    tsne(args, cfg)
