#!/usr/bin/env python3
"""
Create small training datasets by sampling from each source proportionally.
Maintains video-level splitting to prevent data leakage.
"""

import json
import random
import argparse
from collections import defaultdict
from pathlib import Path


def load_json(json_path):
    """Load JSON dataset file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def group_by_source_and_video(data):
    """Group samples by source and video_id."""
    grouped = defaultdict(lambda: defaultdict(list))

    for item in data['data']:
        source = item['source']
        video_id = item.get('video_id', 'unknown')
        grouped[source][video_id].append(item)

    return grouped


def sample_from_source(source_videos, target_count, seed=3407):
    """
    Sample target_count samples from a source.
    Samples at video level first, then frames from selected videos.
    Ensures no duplicates by tracking used samples.
    """
    random.seed(seed)

    # Get all video IDs and collect all samples
    all_samples = []
    for vid, samples in source_videos.items():
        all_samples.extend(samples)

    total_samples = len(all_samples)

    # If target is greater than or equal to available, use all
    if target_count >= total_samples:
        return all_samples

    # Directly sample without replacement to avoid duplicates
    selected_samples = random.sample(all_samples, target_count)

    return selected_samples


def create_small_dataset(input_json_path, output_json_path, target_size=5000, seed=3407):
    """
    Create a smaller dataset by sampling from each source.

    Args:
        input_json_path: Path to original train.json
        output_json_path: Path to save small dataset
        target_size: Total number of samples (will be balanced 50/50 real/fake)
        seed: Random seed for reproducibility
    """
    print(f"\n{'='*60}")
    print(f"Creating small dataset: {target_size} samples")
    print(f"{'='*60}\n")

    # Load data
    data = load_json(input_json_path)
    grouped = group_by_source_and_video(data)

    # Calculate target counts
    target_real = target_size // 2
    target_fake = target_size - target_real

    # Print source statistics
    print("Original source distribution:")
    for source in sorted(grouped.keys()):
        total = sum(len(frames) for frames in grouped[source].values())
        num_videos = len(grouped[source])
        print(f"  {source:20s}: {total:5d} samples from {num_videos:4d} videos")
    print()

    # Separate real and fake sources
    real_sources = {}
    fake_sources = {}

    for source, videos in grouped.items():
        # Check if source is real or fake by looking at first sample
        first_video = next(iter(videos.values()))
        if first_video[0]['label'] == 0:
            real_sources[source] = videos
        else:
            fake_sources[source] = videos

    print(f"Target: {target_real} real + {target_fake} fake = {target_size} total\n")

    # Sample from real sources (proportionally if multiple real sources)
    selected_real = []
    real_total = sum(sum(len(frames) for frames in videos.values())
                     for videos in real_sources.values())

    for source, videos in real_sources.items():
        source_total = sum(len(frames) for frames in videos.values())
        source_target = int(target_real * source_total / real_total)
        sampled = sample_from_source(videos, source_target, seed)
        selected_real.extend(sampled)
        print(f"Real - {source:20s}: sampled {len(sampled):5d}")

    # Adjust if we're short
    if len(selected_real) < target_real:
        shortage = target_real - len(selected_real)
        print(f"\nAdjusting real samples: adding {shortage} more")
        # Sample more from largest real source
        largest_real_source = max(real_sources.keys(),
                                  key=lambda s: sum(len(f) for f in real_sources[s].values()))
        additional = sample_from_source(real_sources[largest_real_source],
                                       shortage, seed + 1)
        selected_real.extend(additional)

    selected_real = selected_real[:target_real]
    print(f"\nTotal real selected: {len(selected_real)}")

    # Sample from fake sources proportionally
    print()
    selected_fake = []
    fake_total = sum(sum(len(frames) for frames in videos.values())
                     for videos in fake_sources.values())

    for source, videos in fake_sources.items():
        source_total = sum(len(frames) for frames in videos.values())
        source_target = int(target_fake * source_total / fake_total)
        sampled = sample_from_source(videos, source_target, seed)
        selected_fake.extend(sampled)
        print(f"Fake - {source:20s}: sampled {len(sampled):5d}")

    # Adjust if we're short
    if len(selected_fake) < target_fake:
        shortage = target_fake - len(selected_fake)
        print(f"\nAdjusting fake samples: adding {shortage} more")
        # Sample more from largest fake source
        largest_fake_source = max(fake_sources.keys(),
                                  key=lambda s: sum(len(f) for f in fake_sources[s].values()))
        additional = sample_from_source(fake_sources[largest_fake_source],
                                       shortage, seed + 1)
        selected_fake.extend(additional)

    selected_fake = selected_fake[:target_fake]
    print(f"\nTotal fake selected: {len(selected_fake)}")

    # Combine and shuffle
    all_samples = selected_real + selected_fake
    random.seed(seed)
    random.shuffle(all_samples)

    # Create output JSON
    output_data = {
        "metadata": {
            "split": "train_small",
            "created_date": data['metadata'].get('created_date', ''),
            "total_samples": len(all_samples),
            "real_count": len(selected_real),
            "fake_count": len(selected_fake),
            "target_size": target_size,
            "seed": seed,
            "sampled_from": str(input_json_path)
        },
        "data": all_samples
    }

    # Save to file
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Small dataset saved to: {output_json_path}")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Real: {len(selected_real)} | Fake: {len(selected_fake)}")
    print(f"{'='*60}\n")

    return output_data


def main():
    parser = argparse.ArgumentParser(description='Create small training datasets')
    parser.add_argument('--input', type=str,
                       default='/data/data/deepfake_finetune_dataset/jsons/train.json',
                       help='Input train.json path')
    parser.add_argument('--output_dir', type=str,
                       default='/data/data/deepfake_finetune_dataset/jsons',
                       help='Output directory for small datasets')
    parser.add_argument('--sizes', type=int, nargs='+',
                       default=[2000, 5000, 8000],
                       help='Dataset sizes to create (default: 2000 5000 8000)')
    parser.add_argument('--seed', type=int, default=3407,
                       help='Random seed (default: 3407)')

    args = parser.parse_args()

    # Create datasets of different sizes
    for size in args.sizes:
        output_path = f"{args.output_dir}/train_small_{size}.json"
        create_small_dataset(args.input, output_path, size, args.seed)


if __name__ == '__main__':
    main()
