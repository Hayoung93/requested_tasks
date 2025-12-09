#!/usr/bin/env python3
"""
Regenerate training JSONs with youtube_real data

This script modifies train.json and train_small_5000.json to include youtube_real frames,
replacing 50% of klleon_real samples with youtube_real for better diversity.

Strategy:
- train_small_5000.json: 1,250 klleon + 1,250 youtube_real
- train.json: 50% of klleon + all youtube_real (1,378)

Usage:
    python scripts/regenerate_train_jsons_with_youtube.py
"""

import json
import glob
import os
import random
from pathlib import Path
from collections import Counter

# Set random seed for reproducibility
SEED = 3407
random.seed(SEED)


def load_json(json_path):
    """Load JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def save_json(data, json_path):
    """Save JSON file"""
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Saved: {json_path}")


def backup_json(json_path):
    """Create backup of original JSON"""
    backup_dir = os.path.join(os.path.dirname(json_path), 'jsons_backup')
    os.makedirs(backup_dir, exist_ok=True)

    filename = os.path.basename(json_path)
    backup_path = os.path.join(backup_dir, filename)

    # Only backup if not already exists
    if not os.path.exists(backup_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        with open(backup_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Backup created: {backup_path}")
    else:
        print(f"  Backup already exists: {backup_path}")


def get_youtube_real_frames():
    """
    Get all youtube_real frame paths

    Returns:
        list: List of dicts with frame information
    """
    frames_dir = '/data/data/deepfake_finetune_dataset/real/youtube_real/frames'
    frame_files = glob.glob(os.path.join(frames_dir, '*.png'))
    frame_files = sorted(frame_files)  # Sort for reproducibility

    frames_data = []
    for frame_path in frame_files:
        filename = os.path.basename(frame_path)
        # Extract video_id from filename: {video_name}_frame_{idx}.png
        video_id = filename.split('_frame_')[0]

        frames_data.append({
            'image_path': frame_path,
            'label': 0,  # Real
            'label_name': 'real',
            'source': 'youtube_real',
            'video_id': video_id
        })

    return frames_data


def regenerate_train_small_5000(original_json_path, output_json_path):
    """
    Regenerate train_small_5000.json with 50% klleon + 50% youtube_real

    Target: 1,250 klleon + 1,250 youtube_real = 2,500 real samples
    """
    print("\n" + "=" * 70)
    print("Regenerating train_small_5000.json")
    print("=" * 70)

    # Load original
    data = load_json(original_json_path)

    # Separate real and fake
    real_samples = [item for item in data['data'] if item['label'] == 0]
    fake_samples = [item for item in data['data'] if item['label'] == 1]

    print(f"Original real samples: {len(real_samples)}")
    print(f"Original fake samples: {len(fake_samples)}")

    # Get klleon_real samples (all current real samples are klleon_real)
    klleon_samples = real_samples.copy()

    # Randomly select 50% of klleon_real
    target_klleon = 1250
    random.shuffle(klleon_samples)
    selected_klleon = klleon_samples[:target_klleon]

    print(f"Selected klleon_real: {len(selected_klleon)}")

    # Get youtube_real frames
    youtube_frames = get_youtube_real_frames()
    print(f"Available youtube_real frames: {len(youtube_frames)}")

    # Select 1,250 youtube_real frames (or all if less)
    target_youtube = 1250
    random.shuffle(youtube_frames)
    selected_youtube = youtube_frames[:min(target_youtube, len(youtube_frames))]

    print(f"Selected youtube_real: {len(selected_youtube)}")

    # Combine new real data
    new_real_samples = selected_klleon + selected_youtube
    random.shuffle(new_real_samples)  # Shuffle combined data

    # Combine with fake data
    new_data = new_real_samples + fake_samples
    random.shuffle(new_data)  # Final shuffle

    # Create new JSON structure
    new_json = {
        'metadata': {
            'split': 'train',
            'total_samples': len(new_data),
            'real_count': len(new_real_samples),
            'fake_count': len(fake_samples),
            'klleon_count': len(selected_klleon),
            'youtube_real_count': len(selected_youtube)
        },
        'data': new_data
    }

    # Save
    save_json(new_json, output_json_path)

    # Print summary
    print(f"\nNew composition:")
    print(f"  Total samples: {len(new_data)}")
    print(f"  Real:  {len(new_real_samples)} (klleon: {len(selected_klleon)}, youtube: {len(selected_youtube)})")
    print(f"  Fake:  {len(fake_samples)}")
    print("=" * 70)


def regenerate_train_full(original_json_path, output_json_path):
    """
    Regenerate train.json with 50% klleon + all youtube_real

    Target: 50% of original klleon + all youtube_real (1,378)
    """
    print("\n" + "=" * 70)
    print("Regenerating train.json")
    print("=" * 70)

    # Load original
    data = load_json(original_json_path)

    # Separate real and fake
    real_samples = [item for item in data['data'] if item['label'] == 0]
    fake_samples = [item for item in data['data'] if item['label'] == 1]

    print(f"Original real samples: {len(real_samples)}")
    print(f"Original fake samples: {len(fake_samples)}")

    # Get klleon_real samples
    klleon_samples = real_samples.copy()

    # Select 50% of klleon_real
    target_klleon = len(klleon_samples) // 2
    random.shuffle(klleon_samples)
    selected_klleon = klleon_samples[:target_klleon]

    print(f"Selected klleon_real (50%): {len(selected_klleon)}")

    # Get ALL youtube_real frames
    youtube_frames = get_youtube_real_frames()
    print(f"Selected youtube_real (all): {len(youtube_frames)}")

    # Combine new real data
    new_real_samples = selected_klleon + youtube_frames
    random.shuffle(new_real_samples)

    # Combine with fake data
    new_data = new_real_samples + fake_samples
    random.shuffle(new_data)

    # Create new JSON structure
    new_json = {
        'metadata': {
            'split': 'train',
            'total_samples': len(new_data),
            'real_count': len(new_real_samples),
            'fake_count': len(fake_samples),
            'klleon_count': len(selected_klleon),
            'youtube_real_count': len(youtube_frames)
        },
        'data': new_data
    }

    # Save
    save_json(new_json, output_json_path)

    # Print summary
    print(f"\nNew composition:")
    print(f"  Total samples: {len(new_data)}")
    print(f"  Real:  {len(new_real_samples)} (klleon: {len(selected_klleon)}, youtube: {len(youtube_frames)})")
    print(f"  Fake:  {len(fake_samples)}")
    print("=" * 70)


def validate_json(json_path):
    """Validate generated JSON"""
    print(f"\nValidating {os.path.basename(json_path)}...")

    data = load_json(json_path)

    # Check metadata
    metadata = data.get('metadata', {})
    print(f"  Total samples: {metadata.get('total_samples', 'N/A')}")
    print(f"  Real count: {metadata.get('real_count', 'N/A')}")
    print(f"  Fake count: {metadata.get('fake_count', 'N/A')}")

    # Check source distribution
    sources = [item.get('source', 'unknown') for item in data['data']]
    source_counts = Counter(sources)

    print("\n  Source distribution:")
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {source:20s}: {count:5d} samples")

    # Check file existence (sample check)
    print("\n  Checking file paths (first 10)...")
    valid_count = 0
    for item in data['data'][:10]:
        if os.path.exists(item['image_path']):
            valid_count += 1
        else:
            print(f"    ⚠ File not found: {item['image_path']}")

    if valid_count == 10:
        print(f"    ✓ All sampled paths exist")
    else:
        print(f"    ⚠ {10 - valid_count} paths not found")


if __name__ == '__main__':
    json_dir = '/data/data/deepfake_finetune_dataset/jsons'

    print("=" * 70)
    print("Training JSON Regeneration with youtube_real")
    print("=" * 70)
    print(f"Random seed: {SEED}")
    print(f"JSON directory: {json_dir}")

    # Backup original JSONs
    print("\nCreating backups...")
    backup_json(os.path.join(json_dir, 'train.json'))
    backup_json(os.path.join(json_dir, 'train_small_5000.json'))

    # Regenerate train_small_5000.json
    regenerate_train_small_5000(
        original_json_path=os.path.join(json_dir, 'train_small_5000.json'),
        output_json_path=os.path.join(json_dir, 'train_small_5000.json')
    )

    # Regenerate train.json
    regenerate_train_full(
        original_json_path=os.path.join(json_dir, 'train.json'),
        output_json_path=os.path.join(json_dir, 'train.json')
    )

    # Validate both
    print("\n" + "=" * 70)
    print("Validation")
    print("=" * 70)
    validate_json(os.path.join(json_dir, 'train_small_5000.json'))
    validate_json(os.path.join(json_dir, 'train.json'))

    print("\n" + "=" * 70)
    print("✓ JSON Regeneration Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review the new JSONs")
    print("  2. Test loading with JSONDataset")
    print("  3. Start training with new data")
    print("=" * 70)
