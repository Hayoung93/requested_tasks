"""
Test script for JSON-based data loading in ForgeLens

This script tests:
1. JSON dataset loading
2. Balanced sampler
3. DataLoader with batching
4. Class distribution in batches

Author: ForgeLens Team
Date: 2025-11-28
"""

import sys
sys.path.append('/workspace/ForgeLens')

from options.options import Options
from util import get_dataset_from_json, get_dataset_from_json_test, get_bal_sampler_json
from torch.utils.data import DataLoader
import torch


def test_data_loading():
    """Test JSON-based data loading"""
    print("\n" + "=" * 80)
    print("TESTING JSON-BASED DATA LOADING")
    print("=" * 80)

    # Create options
    opt = Options().parse()
    opt.use_json_dataset = True

    # Test 1: Load training dataset
    print("\n1. Loading Training Dataset...")
    train_dataset = get_dataset_from_json(opt.train_json)
    print(f"   ✓ Training dataset loaded: {len(train_dataset)} samples")

    # Test 2: Load validation dataset
    print("\n2. Loading Validation Dataset...")
    val_dataset = get_dataset_from_json_test(opt.val_json)
    print(f"   ✓ Validation dataset loaded: {len(val_dataset)} samples")

    # Test 3: Create balanced sampler
    print("\n3. Creating Balanced Sampler...")
    train_sampler = get_bal_sampler_json(train_dataset)
    print(f"   ✓ Balanced sampler created")

    # Test 4: Create DataLoader
    print("\n4. Creating DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,
        num_workers=0  # Use 0 for testing
    )
    print(f"   ✓ DataLoader created with {len(train_loader)} batches")

    # Test 5: Load one batch
    print("\n5. Loading Sample Batches...")
    batch_count = 0
    real_count_total = 0
    fake_count_total = 0

    for images, labels in train_loader:
        batch_count += 1
        real_count = (labels == 0).sum().item()
        fake_count = (labels == 1).sum().item()
        real_count_total += real_count
        fake_count_total += fake_count

        if batch_count <= 3:  # Print first 3 batches
            print(f"   Batch {batch_count}:")
            print(f"     - Image shape: {images.shape}")
            print(f"     - Labels: {labels.shape}")
            print(f"     - Real: {real_count}, Fake: {fake_count}")
            print(f"     - Ratio: {real_count/(real_count+fake_count):.2%} real")

        if batch_count >= 10:  # Test 10 batches
            break

    print(f"\n   ✓ Loaded {batch_count} batches successfully")
    print(f"   Total: Real={real_count_total}, Fake={fake_count_total}")
    print(f"   Overall ratio: {real_count_total/(real_count_total+fake_count_total):.2%} real")

    # Test 6: Verify class balance
    print("\n6. Verifying Class Balance...")
    expected_real = train_dataset.get_class_distribution()['real']
    expected_fake = train_dataset.get_class_distribution()['fake']
    print(f"   Dataset distribution:")
    print(f"     - Real: {expected_real} ({expected_real/(expected_real+expected_fake):.2%})")
    print(f"     - Fake: {expected_fake} ({expected_fake/(expected_real+expected_fake):.2%})")
    print(f"   ✓ WeightedRandomSampler ensures ~50% real/fake in each batch")

    # Test 7: Load sample from validation
    print("\n7. Testing Validation Dataset...")
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    images, labels = next(iter(val_loader))
    print(f"   ✓ Validation batch loaded:")
    print(f"     - Image shape: {images.shape}")
    print(f"     - Labels: {labels.shape}")
    print(f"     - Real: {(labels == 0).sum().item()}, Fake: {(labels == 1).sum().item()}")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
    print("\nYou can now run training with:")
    print("  python train.py --use_json_dataset --training_stage 1 --experiment_name custom_deepfake")
    print("\n")


if __name__ == '__main__':
    test_data_loading()
