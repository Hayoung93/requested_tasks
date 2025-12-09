#!/usr/bin/env python3
"""
Analyze misclassified samples from evaluation results.
"""

import json
import argparse
from collections import Counter


def analyze_misclassified(json_path):
    """Analyze misclassified samples JSON file."""

    with open(json_path, 'r') as f:
        data = json.load(f)

    print("="*70)
    print("Misclassified Samples Analysis")
    print("="*70)
    print(f"File: {json_path}")
    print()

    # Summary
    print("Summary:")
    print("-"*70)
    print(f"Total misclassified: {data['total_misclassified']}")
    print(f"False Positives (Real predicted as Fake): {data['false_positives']}")
    print(f"False Negatives (Fake predicted as Real): {data['false_negatives']}")
    print()

    # Analyze False Positives
    if data['false_positives'] > 0:
        print("="*70)
        print("FALSE POSITIVES (Real images predicted as Fake)")
        print("="*70)

        fp_samples = data['false_positive_samples']

        # Group by source
        sources = Counter([s['source'] for s in fp_samples])
        print(f"\nBy Source:")
        for source, count in sources.most_common():
            print(f"  {source}: {count}")

        # Group by video
        videos = Counter([s['video_id'] for s in fp_samples])
        print(f"\nTop 10 Videos with most FP:")
        for video_id, count in videos.most_common(10):
            print(f"  {video_id}: {count} frames")

        # Show samples with highest confidence (most confident wrong predictions)
        print(f"\nTop 10 Most Confident False Positives:")
        sorted_fp = sorted(fp_samples, key=lambda x: x['confidence'], reverse=True)
        for i, sample in enumerate(sorted_fp[:10], 1):
            print(f"\n{i}. Confidence: {sample['confidence']:.4f}")
            print(f"   Path: {sample['image_path']}")
            print(f"   Source: {sample['source']}")
            print(f"   Video: {sample['video_id']}")

    # Analyze False Negatives
    if data['false_negatives'] > 0:
        print("\n" + "="*70)
        print("FALSE NEGATIVES (Fake images predicted as Real)")
        print("="*70)

        fn_samples = data['false_negative_samples']

        # Group by source
        sources = Counter([s['source'] for s in fn_samples])
        print(f"\nBy Source:")
        for source, count in sources.most_common():
            print(f"  {source}: {count}")

        # Group by video
        videos = Counter([s['video_id'] for s in fn_samples])
        print(f"\nTop 10 Videos with most FN:")
        for video_id, count in videos.most_common(10):
            print(f"  {video_id}: {count} frames")

        # Show samples with lowest confidence (most confident wrong predictions)
        print(f"\nTop 10 Most Confident False Negatives:")
        sorted_fn = sorted(fn_samples, key=lambda x: x['confidence'])
        for i, sample in enumerate(sorted_fn[:10], 1):
            print(f"\n{i}. Confidence: {sample['confidence']:.4f}")
            print(f"   Path: {sample['image_path']}")
            print(f"   Source: {sample['source']}")
            print(f"   Video: {sample['video_id']}")

    print("\n" + "="*70)
    print()

    return data


def export_file_list(json_path, output_path, error_type='all'):
    """Export list of misclassified file paths."""

    with open(json_path, 'r') as f:
        data = json.load(f)

    if error_type == 'all':
        samples = data['false_positive_samples'] + data['false_negative_samples']
    elif error_type == 'fp':
        samples = data['false_positive_samples']
    elif error_type == 'fn':
        samples = data['false_negative_samples']
    else:
        raise ValueError(f"Unknown error_type: {error_type}")

    file_paths = [s['image_path'] for s in samples]

    with open(output_path, 'w') as f:
        for path in file_paths:
            f.write(path + '\n')

    print(f"✓ Exported {len(file_paths)} file paths to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze misclassified samples')
    parser.add_argument('--json', type=str, required=True,
                       help='Path to misclassified_samples.json')
    parser.add_argument('--export', type=str, default='',
                       help='Export file paths to text file')
    parser.add_argument('--error_type', type=str, default='all',
                       choices=['all', 'fp', 'fn'],
                       help='Type of errors to export (all, fp=false positives, fn=false negatives)')

    args = parser.parse_args()

    # Analyze
    analyze_misclassified(args.json)

    # Export if requested
    if args.export:
        export_file_list(args.json, args.export, args.error_type)
