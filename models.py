import torch
import torch.nn as nn
from torchvision import models
from mambavision import create_model as create_mamba


def get_models(cfg):
    if cfg.model.version == "v1":
        model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
        model.head = torch.nn.Linear(model.head.in_features, cfg.data.num_classes)
    elif cfg.model.version == "v2":
        model = create_mamba("mamba_vision_B", in_chans=cfg.data.in_channel)
        cp = torch.load("/workspace/requested_tasks/mambavision_base_1k.pth.tar")
        if cfg.data.in_channel == 3:
            model.load_state_dict(cp["state_dict"])
        elif cfg.data.in_channel == 4:
            state_dict = cp["state_dict"]
            state_dict = {k: v for k, v in state_dict.items() if k != "patch_embed.conv_down.0.weight"}
            model.load_state_dict(state_dict, strict=False)
        model.head = torch.nn.Linear(model.head.in_features, cfg.data.num_classes)
    elif cfg.model.version == "v3":
        import sys
        sys.path.append("/workspace/efficient-kan/src")
        from efficient_kan import KAN
        model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
        clssifier = KAN([model.head.in_features, 128, cfg.data.num_classes])
        model.head = clssifier
    elif cfg.model.version == "v4":
        model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        model.head = torch.nn.Linear(model.head.in_features, cfg.data.num_classes)
    else:
        raise Exception("Not supported model version: {}".format(cfg.model.version))
    return model


class ContrastiveModel(nn.Module):
    def __init__(self, cfg, model, criterion, criterion_cont):
        super(ContrastiveModel, self).__init__()
        self.cfg = cfg
        self.model = model
        assert isinstance(model, models.SwinTransformer), "Currently only SwinTransformer is supported"
        self.criterion = criterion
        self.criterion_cont = criterion_cont

        head = nn.Linear(model.head.in_features, cfg.data.num_classes)
        self.model.head = head
    
    def forward(self, x, labels):
        feat = self.model.features(x)
        out = self.model.norm(feat)
        out = self.model.permute(out)
        out = self.model.avgpool(out)
        out = self.model.flatten(out)
        out = self.model.head(out)
        losses = {"class": self.criterion(out, labels)}
        if self.training:
            loss_cont = self.criterion_cont(feat.mean(dim=(1, 2)), labels)
            losses["contrastive"] = loss_cont
        return out, losses
