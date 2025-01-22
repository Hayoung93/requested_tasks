import torch.nn as nn
from torchvision import models
from timm import models as tmodels


def get_model(cfg, criterions):
    model = DeepfakeClassificationModel(cfg, criterions)
    return model


class DeepfakeClassificationModel(nn.Module):
    def __init__(self, cfg, criterions):
        super().__init__()
        self.cfg = cfg
        self.criterions = criterions

        if cfg.model.version == "v1":
            backbone = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
            num_last_feat = backbone.head.in_features
            backbone.head = nn.Identity()
            # backbone.head = nn.Linear(backbone.head.in_features, cfg.data.num_classes)
            head_class = nn.Linear(num_last_feat, cfg.data.num_classes)
        elif cfg.model.version == "v2":
            backbone = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
            num_last_feat = backbone.head.in_features
            backbone.head = nn.Identity()
            head_class = nn.Linear(num_last_feat, cfg.data.num_classes)
        elif cfg.model.version == "v3":
            backbone = tmodels.swin_transformer.swin_tiny_patch4_window7_224(pretrained=True)
            num_last_feat = backbone.head.fc.in_features
            backbone.head.fc = nn.Identity()
            head_class = nn.Linear(num_last_feat, cfg.data.num_classes)
        else:
            raise Exception("Not supported model version: {}".format(cfg.model.version))
        self.backbone = backbone
        self.head_class = head_class
    
    def forward(self, x, labels):
        feat = self.backbone(x) 
        logit = self.head_class(feat)
        losses = {"class": sum([criterion(logit, labels) for criterion in self.criterions])}
        return logit, losses
