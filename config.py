import argparse
from yacs.config import CfgNode as CN

_C = CN()

_C.data = CN()
_C.data.root           = ("/data/mnt_ssd/FaceForensics_origin", "root")
_C.data.quality        = ("c40"                               , "quality")
_C.data.num_classes    = (2                                   , "num_classes")
_C.data.in_channel     = (3                                   , "in_channel")
_C.data.input_size     = (224                                 , "input_size")
_C.data.curriculum     = (False                               , "curriculum")

_C.io = CN()
_C.io.save_dir = ("results"  , "save_dir")
_C.io.exp_name = ("debug"    , "exp_name")
_C.io.resume   = (""         , "resume")
_C.io.pretrain = (""         , "pretrain")

_C.run = CN()
_C.run.batch_size    = (64                 , "batch_size")
_C.run.num_workers   = (16                 , "num_workers")
_C.run.optimizer     = ("AdamW"            , "optimizer")
_C.run.criterion     = ("ce"               , "criterion")
_C.run.scheduler     = ("CosineAnnealingLR", "scheduler")
_C.run.lr            = (1e-4               , "lr")
_C.run.weight_decay  = (1e-6               , "weight_decay")
_C.run.epochs        = (100                , "epochs")
_C.run.parallel      = ("DP"               , "parallel")
_C.run.val_interval  = (1                  , "val_interval")

_C.model = CN()
_C.model.version  = ("v1"             , "model_version")

_C.curriculum = CN()
_C.curriculum.milestones            = ([20, 70]      , "milestones")
_C.curriculum.load_difficulty       = (""            , "load_difficulty")
_C.curriculum.difficulty_function   = ("difficulty-1", "difficulty_function")
_C.curriculum.pace_function         = ("pace-1"      , "pace_function")
_C.curriculum.curriculum_pretrained = ("/workspace/SBI_Deepfake/results/20250207v1_swin-t/checkpoint.pth", "curriculum_pretrained")

# MISC parameters - no args for this field
_C.misc = CN()

def get_cfg_defaults():
    return _C.clone()

def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--quality", type=str)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--in_channel", type=int)
    parser.add_argument("--input_size", type=int)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--pretrain", type=str)
    parser.add_argument("--new_spacing", nargs="+", type=float)
    parser.add_argument("--target_size", nargs="+", type=int)
    parser.add_argument("--train_slice_num", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--criterion", type=str)
    parser.add_argument("--scheduler", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--parallel", type=str)
    parser.add_argument("--val_interval", type=int)
    parser.add_argument("--model_version", type=str)
    parser.add_argument("--milestones", nargs="+", type=int)
    parser.add_argument("--load_difficulty", type=str)
    parser.add_argument("--difficulty_function", type=str)
    parser.add_argument("--pace_function", type=str)
    parser.add_argument("--curriculum_pretrained", type=str)

    parser.add_argument("--eval", action="store_true")

    args, _ = parser.parse_known_args()
    cfg = get_cfg_defaults()
    cfg = merge_args(args, cfg)
    args = set_args_with_cfg(args, cfg)  # work only when last key of cfg matches args
    return args, cfg


def set_args_with_cfg(args, cfg):
    # work only when last key of cfg matches args
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
