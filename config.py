import argparse
from yacs.config import CfgNode as CN

_C = CN()

_C.data = CN()
_C.data.root           = ("/data/mnt_ssd/", "root")
_C.data.dataset_name   = ("FF"            , "dataset_name")
_C.data.num_classes    = (2               , "num_classes")
_C.data.in_channel     = (3               , "in_channel")
_C.data.image_size     = (224             , "image_size")

_C.io = CN()
_C.io.save_dir = ("results"  , "save_dir")
_C.io.exp_name = ("debug"    , "exp_name")
_C.io.resume   = (""         , "resume")
_C.io.pretrain = (""         , "pretrain")

_C.run = CN()
_C.run.batch_size    = (32         , "batch_size")
_C.run.num_workers   = (2          , "num_workers")
_C.run.optimizer     = ("SAM"      , "optimizer")
_C.run.criterion     = ("CE"       , "criterion")
_C.run.scheduler     = ("custom_v1", "scheduler")
_C.run.lr            = (1e-3       , "lr")
_C.run.momentum      = (0.9        , "momentum")
_C.run.weight_decay  = (1e-6       , "weight_decay")
_C.run.epochs        = (100        , "epochs")
_C.run.val_interval  = (5          , "val_interval")

_C.model = CN()
_C.model.backbone = ("swin-v1-tiny", "backbone")
# parameters for swin-v1-base model - no args for this field
_C.model.DATA = CN()
_C.model.DATA.IMG_SIZE = (224, "img_size")
_C.model.TRAIN = CN()
_C.model.TRAIN.USE_CHECKPOINT = (False, "use_checkpoint")
_C.model.MODEL = CN()
_C.model.MODEL.NUM_CLASSES = (2, "n_classes")
_C.model.MODEL.DROP_RATE = (0.0, "drop_rate")
_C.model.MODEL.DROP_PATH_RATE = (0.5, "drop_path_rate")
_C.model.MODEL.SWIN = CN()
_C.model.MODEL.SWIN.PATCH_SIZE = (4, "patch_size")
_C.model.MODEL.SWIN.IN_CHANS = (3, "in_chans")
_C.model.MODEL.SWIN.EMBED_DIM = (128, "embed_dim")
_C.model.MODEL.SWIN.DEPTHS = ([2, 2, 18, 2], "depths")
_C.model.MODEL.SWIN.NUM_HEADS = ([4, 8, 16, 32], "num_heads")
_C.model.MODEL.SWIN.WINDOW_SIZE = (7, "window_size")
_C.model.MODEL.SWIN.MLP_RATIO = (4.0, "mlp_ratio")
_C.model.MODEL.SWIN.QKV_BIAS = (True, "qkv_bias")
_C.model.MODEL.SWIN.QK_SCALE = (None, "qk_scale")
_C.model.MODEL.SWIN.APE = (False, "ape")
_C.model.MODEL.SWIN.PATCH_NORM = (True, "patch_norm")

# MISC parameters - no args for this field
_C.misc = CN()

def get_cfg_defaults():
    return _C.clone()

def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--in_channel", type=int)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--pretrain", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--criterion", type=str)
    parser.add_argument("--scheduler", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--val_interval", type=int)
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--image_size", type=int)

    parser.add_argument("--eval", action="store_true")

    args, _ = parser.parse_known_args()
    cfg = get_cfg_defaults()
    cfg = merge_args(args, cfg)
    args = set_args_with_cfg(args, cfg)
    return args, cfg


def set_args_with_cfg(args, cfg):
    cfg_keys = list(get_cfg_keys(cfg, cfg.keys()))
    for k in cfg_keys:
        _k = k.split(".")[-1]
        if hasattr(args, _k) and (getattr(args, _k) is None):
            setattr(args, _k, eval("cfg." + k))
    return args


def merge_args(args, cfg):
    new_cfg = CN({})
    new_cfg.set_new_allowed(True)
    key_gen = get_cfg_keys(cfg, cfg.keys())
    while True:
        try:
            key = next(key_gen)
            value, args_key = eval("cfg." + key)
            if (args_key in args) and (eval("args." + args_key) is not None):
                value = eval("args." + args_key)
            key_split =  key.split(".")
            t1 = {key_split[-1]: value}
            t2 = {}
            for k in key_split[:-1][::-1]:
                if t1 == {}:
                    t1[k] = t2
                    t2 = {}
                else:
                    t2[k] = t1
                    t1 = {}
            if t1 == {}:
                t2 = CN(t2)
                new_cfg = merge_cfg(t2, new_cfg)
            else:
                t1 = CN(t1)
                new_cfg = merge_cfg(t1, new_cfg)
        except StopIteration:
            break
    return new_cfg


def get_cfg_keys(cn, keys):
    for key in keys:
        cur_node = eval("cn." + key)
        if type(cur_node) == CN:
            yield from get_cfg_keys(cn, list(map(lambda x: key + "." + x, cur_node.keys())))
        else:
            yield key

def merge_cfg(a, b):
    for k, v in a.items():
        if k in b:
            if isinstance(v, CN):
                merge_cfg(v, b[k])
            else:
                b[k] = v
        else:
            b[k] = v
    return b
