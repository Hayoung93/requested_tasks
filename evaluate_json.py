"""
Evaluation script for JSON-based datasets

This script evaluates ForgeLens models (Stage 1 or Stage 2) on JSON-based test datasets.

Usage:
    python evaluate_json.py --eval_stage 1 --weights <path_to_model> --test_json <path_to_test_json>

Author: ForgeLens Team
Date: 2025-12-02
"""

import os
import torch
import numpy as np
import random
from tqdm import tqdm

from models.network.net_stage1 import net_stage1
from models.network.net_stage2 import net_stage2
from options.options import Options
from util import Logger, get_dataset_from_json_test
from sklearn.metrics import accuracy_score, average_precision_score
from torch.amp import autocast
from torch.cuda.amp import GradScaler


def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate_on_json(model, test_loader, eval_stage, device='cuda'):
    """Evaluate model on a test loader"""
    model.eval()

    all_targets = []
    all_pre_probs = []

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)

            with autocast('cuda'):
                if eval_stage == 1:
                    pre, _ = model(data)
                else:
                    pre = model(data)

                pre_prob = torch.sigmoid(pre).cpu()
                target = target.cpu()

                all_targets.extend(target.numpy())
                all_pre_probs.extend(pre_prob.numpy())

    # Calculate metrics
    all_targets = np.array(all_targets)
    all_pre_probs = np.array(all_pre_probs)

    acc = accuracy_score(all_targets, all_pre_probs > 0.5)
    ap = average_precision_score(all_targets, all_pre_probs)

    return acc, ap, all_targets, all_pre_probs


if __name__ == '__main__':
    seed_torch(3407)

    # Options
    options = Options()
    opt = options.parse()

    # Create log directory
    log_dir = os.path.join('./check_points', opt.experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    # Logger
    log_filename = f'evaluation_stage{opt.eval_stage}_json.log'
    Logger(os.path.join(log_dir, log_filename))

    print("=" * 80)
    print(f"EVALUATING STAGE {opt.eval_stage} MODEL ON JSON DATASET")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from: {opt.weights}")
    if opt.eval_stage == 1:
        model = net_stage1()
    else:
        model = net_stage2(opt, train=False)

    model_load = torch.load(opt.weights)
    model.load_state_dict(model_load['model_state_dict'])

    # Print model checkpoint information
    print("\nModel Checkpoint Information:")
    if 'epoch' in model_load:
        print(f"  Epoch: {model_load['epoch']}")
    if 'train_loss' in model_load:
        print(f"  Train loss: {model_load['train_loss']:.4f}")
    if 'val_loss' in model_load:
        print(f"  Validation loss: {model_load['val_loss']:.4f}")
    if 'val_acc' in model_load:
        print(f"  Validation accuracy: {model_load['val_acc'] * 100:.2f}%")
    if 'val_ap' in model_load:
        print(f"  Validation AP: {model_load['val_ap'] * 100:.2f}%")
    if 'best_val_loss' in model_load:
        print(f"  Best validation loss: {model_load['best_val_loss']:.4f}")
    if 'best_val_acc' in model_load:
        print(f"  Best validation accuracy: {model_load['best_val_acc'] * 100:.2f}%")

    # Freeze all parameters except fc layer
    for name, p in model.named_parameters():
        if name == "fc.weight" or name == "fc.bias":
            p.requires_grad = True
        else:
            p.requires_grad = False

    model.cuda()
    model.eval()
    print("Model loaded successfully!")

    # Load test dataset
    print(f"\nLoading test dataset from: {opt.test_json}")
    if opt.use_resize_only:
        print(f"Using resize-only mode (nearest neighbor interpolation)")
    else:
        print(f"Using center crop mode")
    test_dataset = get_dataset_from_json_test(opt.test_json, opt)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers
    )

    # Evaluate
    print(f"\nEvaluating on {len(test_dataset)} samples...")
    acc, ap, targets, probs = evaluate_on_json(model, test_loader, opt.eval_stage)

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Test JSON: {opt.test_json}")
    print(f"Total samples: {len(targets)}")
    print(f"Real samples: {sum(targets == 0)}")
    print(f"Fake samples: {sum(targets == 1)}")
    print(f"\nOverall Accuracy (ACC): {acc * 100:.2f}%")
    print(f"Overall Average Precision (AP): {ap * 100:.2f}%")
    print("=" * 80)

    # Class-wise evaluation
    print("\n" + "=" * 80)
    print("CLASS-WISE EVALUATION")
    print("=" * 80)

    # Real class (label 0)
    real_mask = targets == 0
    real_targets = targets[real_mask]
    real_probs = probs[real_mask]
    if len(real_targets) > 0:
        real_acc = accuracy_score(real_targets, real_probs > 0.5)
        real_ap = average_precision_score(real_targets, real_probs)
        print(f"\nReal (Label 0):")
        print(f"  Samples: {len(real_targets)}")
        print(f"  ACC: {real_acc * 100:.2f}%")
        print(f"  AP: {real_ap * 100:.2f}%")

    # Fake class (label 1)
    fake_mask = targets == 1
    fake_targets = targets[fake_mask]
    fake_probs = probs[fake_mask]
    if len(fake_targets) > 0:
        fake_acc = accuracy_score(fake_targets, fake_probs > 0.5)
        fake_ap = average_precision_score(fake_targets, fake_probs)
        print(f"\nFake (Label 1):")
        print(f"  Samples: {len(fake_targets)}")
        print(f"  ACC: {fake_acc * 100:.2f}%")
        print(f"  AP: {fake_ap * 100:.2f}%")

    print("=" * 80)

    # Per-source evaluation if metadata available
    if hasattr(test_dataset, 'data'):
        print("\n" + "=" * 80)
        print("PER-SOURCE EVALUATION")
        print("=" * 80)

        from collections import defaultdict
        source_data = defaultdict(lambda: {'targets': [], 'probs': []})

        for i, item in enumerate(test_dataset.data):
            source = item.get('source', 'unknown')
            source_data[source]['targets'].append(targets[i])
            source_data[source]['probs'].append(probs[i])

        for source in sorted(source_data.keys()):
            src_targets = np.array(source_data[source]['targets'])
            src_probs = np.array(source_data[source]['probs'])

            if len(src_targets) > 0:
                src_acc = accuracy_score(src_targets, src_probs > 0.5)
                src_ap = average_precision_score(src_targets, src_probs)

                print(f"\n{source}:")
                print(f"  Samples: {len(src_targets)}")
                print(f"  Real: {sum(src_targets == 0)}, Fake: {sum(src_targets == 1)}")
                print(f"  ACC: {src_acc * 100:.2f}%")
                print(f"  AP: {src_ap * 100:.2f}%")

        print("=" * 80)
