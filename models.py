import torch
from torchvision import models


def get_models(cfg):
    if cfg.model.version == "v1":
        model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
        model.head = torch.nn.Linear(model.head.in_features, cfg.data.num_classes)
    else:
        raise Exception("Not supported model version: {}".format(cfg.model.version))
    return model
