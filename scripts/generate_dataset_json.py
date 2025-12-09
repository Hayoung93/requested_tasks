"""
Dataset JSON Generation Script for ForgeLens Custom Dataset

This script generates JSON files for train/val/test splits:
- Fake class: 8:1:1 split (video-level, per source)
- Real class: 8:2 split for train/val, real_test for test
- Ensures frames from same video stay in same split

Author: ForgeLens Team
Date: 2025-11-28
"""

import os
import json
import glob
import random
from pathlib import Path
from collections import defaultdict
from datetime import datetime


class DatasetJSONGenerator:
    def __init__(self, dataset_root='/data/data/deepfake_finetune_dataset',
                 output_dir='/data/data/deepfake_finetune_dataset/jsons',
                 seed=3407):
        self.dataset_root = dataset_root
        self.output_dir = output_dir
        self.seed = seed
        random.seed(seed)

        # Initialize splits
        self.splits = {
            'train': [],
            'val': [],
            'test': []
        }

        # Statistics
        self.stats = {
            'train': {'real': 0, 'fake': 0},
            'val': {'real': 0, 'fake': 0},
            'test': {'real': 0, 'fake': 0}
        }

    def collect_fake_frames(self):
        """Collect all fake frames grouped by video and source"""
        print("\n" + "=" * 80)
        print("COLLECTING FAKE FRAMES")
        print("=" * 80)

        fake_data = {
            'kling2.5': defaultdict(list),
            'veo3': defaultdict(list),
            'nanobanana': defaultdict(list),
            'sora': defaultdict(list)
        }

        # 1. kling2.5 dataset
        kling_dir = os.path.join(self.dataset_root, 'kling2.5_dataset')
        if os.path.exists(kling_dir):
            video_dirs = [d for d in glob.glob(os.path.join(kling_dir, '*')) if os.path.isdir(d)]
            for video_dir in video_dirs:
                video_id = os.path.basename(video_dir)
                frames = glob.glob(os.path.join(video_dir, 'frames', '*.png'))
                if frames:
                    fake_data['kling2.5'][video_id] = frames
            print(f"kling2.5: {len(fake_data['kling2.5'])} videos, "
                  f"{sum(len(f) for f in fake_data['kling2.5'].values())} frames")

        # 2. veo3 dataset
        veo_dir = os.path.join(self.dataset_root, 'veo3_dataset')
        if os.path.exists(veo_dir):
            video_dirs = [d for d in glob.glob(os.path.join(veo_dir, '*')) if os.path.isdir(d)]
            for video_dir in video_dirs:
                video_id = os.path.basename(video_dir)
                frames = glob.glob(os.path.join(video_dir, 'frames', '*.png'))
                if frames:
                    fake_data['veo3'][video_id] = frames
            print(f"veo3: {len(fake_data['veo3'])} videos, "
                  f"{sum(len(f) for f in fake_data['veo3'].values())} frames")

        # 3. nanobanana dataset (flat images, treat each as separate "video")
        nano_dir = os.path.join(self.dataset_root, 'nanobanana_dataset')
        if os.path.exists(nano_dir):
            frames = glob.glob(os.path.join(nano_dir, '*.png'))
            for frame_path in frames:
                frame_id = Path(frame_path).stem
                fake_data['nanobanana'][frame_id] = [frame_path]
            print(f"nanobanana: {len(fake_data['nanobanana'])} images")

        # 4. sora dataset (extracted frames)
        sora_dir = os.path.join(self.dataset_root, 'sora_dataset', 'frames')
        if os.path.exists(sora_dir):
            # Group frames by video name
            all_frames = glob.glob(os.path.join(sora_dir, '*.png'))
            for frame_path in all_frames:
                frame_name = Path(frame_path).stem
                # Extract video name (format: {video_name}_frame_{idx})
                video_name = '_'.join(frame_name.split('_')[:-2])  # Remove '_frame_{idx}'
                fake_data['sora'][video_name].append(frame_path)
            print(f"sora: {len(fake_data['sora'])} videos, "
                  f"{sum(len(f) for f in fake_data['sora'].values())} frames")

        return fake_data

    def split_fake_data(self, fake_data):
        """Split fake data 8:1:1 by video (per source)"""
        print("\n" + "=" * 80)
        print("SPLITTING FAKE DATA (8:1:1)")
        print("=" * 80)

        for source, videos_dict in fake_data.items():
            video_ids = list(videos_dict.keys())
            random.shuffle(video_ids)

            n = len(video_ids)
            train_split = int(0.8 * n)
            val_split = int(0.9 * n)

            train_videos = video_ids[:train_split]
            val_videos = video_ids[train_split:val_split]
            test_videos = video_ids[val_split:]

            # Add to splits
            for video_id in train_videos:
                for frame_path in videos_dict[video_id]:
                    self.splits['train'].append({
                        'image_path': frame_path,
                        'label': 1,
                        'label_name': 'fake',
                        'source': source,
                        'video_id': video_id
                    })
                    self.stats['train']['fake'] += 1

            for video_id in val_videos:
                for frame_path in videos_dict[video_id]:
                    self.splits['val'].append({
                        'image_path': frame_path,
                        'label': 1,
                        'label_name': 'fake',
                        'source': source,
                        'video_id': video_id
                    })
                    self.stats['val']['fake'] += 1

            for video_id in test_videos:
                for frame_path in videos_dict[video_id]:
                    self.splits['test'].append({
                        'image_path': frame_path,
                        'label': 1,
                        'label_name': 'fake',
                        'source': source,
                        'video_id': video_id
                    })
                    self.stats['test']['fake'] += 1

            print(f"{source}:")
            print(f"  Train: {len(train_videos)} videos, "
                  f"{sum(len(videos_dict[v]) for v in train_videos)} frames")
            print(f"  Val: {len(val_videos)} videos, "
                  f"{sum(len(videos_dict[v]) for v in val_videos)} frames")
            print(f"  Test: {len(test_videos)} videos, "
                  f"{sum(len(videos_dict[v]) for v in test_videos)} frames")

    def collect_real_test_videos(self):
        """Collect frames from real_test/0_video/ (previously missing)"""
        print("\n" + "=" * 80)
        print("COLLECTING REAL TEST VIDEO FRAMES (0_video/)")
        print("=" * 80)

        real_test_video_frames = []

        video_dir = os.path.join(self.dataset_root, 'real_test', '0_video', '0_real')
        if os.path.exists(video_dir):
            # Find all frames/ directories
            frame_dirs = glob.glob(os.path.join(video_dir, '**', 'frames'), recursive=True)

            for frame_dir in frame_dirs:
                frames = glob.glob(os.path.join(frame_dir, '*.png'))

                # Extract source from path
                rel_path = os.path.relpath(frame_dir, video_dir)
                source = rel_path.replace('/frames', '').replace('/', '_')

                for frame_path in frames:
                    # Extract video_id from filename
                    frame_name = Path(frame_path).stem
                    # Format: {video_stem}_frame_{original_idx:06d}
                    # Remove '_frame_{idx}' to get video_id
                    video_id = '_'.join(frame_name.split('_')[:-2])

                    real_test_video_frames.append({
                        'image_path': frame_path,
                        'label': 0,
                        'label_name': 'real',
                        'source': f'real_test_0_video_{source}',
                        'video_id': video_id
                    })

            # Print per-source statistics
            source_stats = defaultdict(int)
            for frame_data in real_test_video_frames:
                source_stats[frame_data['source']] += 1

            print(f"real_test/0_video/: {len(real_test_video_frames)} total frames")
            for source, count in sorted(source_stats.items()):
                print(f"  {source}: {count} frames")

        return real_test_video_frames

    def collect_real_frames(self):
        """Collect all real frames grouped by video"""
        print("\n" + "=" * 80)
        print("COLLECTING REAL FRAMES")
        print("=" * 80)

        real_train_val = defaultdict(list)
        real_test = []

        # 1. real/ folder (for train/val)
        real_dir = os.path.join(self.dataset_root, 'real', '1. klleon_real_dataset')
        if os.path.exists(real_dir):
            # Find all frames/ directories
            frame_dirs = glob.glob(os.path.join(real_dir, '**', 'frames'), recursive=True)
            for frame_dir in frame_dirs:
                frames = glob.glob(os.path.join(frame_dir, '*.png'))
                if frames:
                    # Use parent directory path as video ID
                    video_path = os.path.dirname(frame_dir)
                    video_id = os.path.relpath(video_path, real_dir)
                    real_train_val[video_id] = frames

            total_frames = sum(len(f) for f in real_train_val.values())
            print(f"real/ (train/val): {len(real_train_val)} videos, {total_frames} frames")

        # 2. real_test/ folder (for test) - excludes 0_video/ which is handled separately
        real_test_dir = os.path.join(self.dataset_root, 'real_test')
        if os.path.exists(real_test_dir):
            # Collect all images recursively
            all_real_test = glob.glob(os.path.join(real_test_dir, '**', '*.png'), recursive=True)
            all_real_test += glob.glob(os.path.join(real_test_dir, '**', '*.jpg'), recursive=True)
            all_real_test += glob.glob(os.path.join(real_test_dir, '**', '*.jpeg'), recursive=True)

            # Exclude images in directories ending with 'fake' (e.g., /1_fake/, /01_fake/, /car_fake/)
            # Also exclude images from 0_video/0_real/*/frames/ (handled separately)
            # Keep images in directories with 'real' or without 'fake'
            real_test_filtered = []
            for img in all_real_test:
                path_parts = img.split(os.sep)
                # Exclude if any directory component ends with '_fake' or is exactly 'fake'
                exclude = False
                for part in path_parts[:-1]:  # Exclude last part (filename)
                    part_lower = part.lower()
                    if part_lower.endswith('_fake') or part_lower == 'fake':
                        exclude = True
                        break

                # Exclude if path contains '0_video/0_real/*/frames/'
                if '0_video' in path_parts and '0_real' in path_parts and 'frames' in path_parts:
                    exclude = True

                if not exclude:
                    real_test_filtered.append(img)

            real_test = real_test_filtered
            excluded_count = len(all_real_test) - len(real_test)

            print(f"real_test/ (images only): {len(all_real_test)} total images")
            print(f"  Excluded (fake directories or 0_video frames): {excluded_count}")
            print(f"  Included (real images only): {len(real_test)}")

        return real_train_val, real_test

    def split_real_data(self, real_train_val, real_test):
        """Split real data 8:2 for train/val, use real_test for test"""
        print("\n" + "=" * 80)
        print("SPLITTING REAL DATA (8:2 for train/val)")
        print("=" * 80)

        # Split train/val by video
        video_ids = list(real_train_val.keys())
        random.shuffle(video_ids)

        n = len(video_ids)
        train_split = int(0.8 * n)

        train_videos = video_ids[:train_split]
        val_videos = video_ids[train_split:]

        # Add to splits
        for video_id in train_videos:
            for frame_path in real_train_val[video_id]:
                self.splits['train'].append({
                    'image_path': frame_path,
                    'label': 0,
                    'label_name': 'real',
                    'source': 'klleon_real',
                    'video_id': video_id
                })
                self.stats['train']['real'] += 1

        for video_id in val_videos:
            for frame_path in real_train_val[video_id]:
                self.splits['val'].append({
                    'image_path': frame_path,
                    'label': 0,
                    'label_name': 'real',
                    'source': 'klleon_real',
                    'video_id': video_id
                })
                self.stats['val']['real'] += 1

        # Add real_test images to test split
        for frame_path in real_test:
            # Extract source from path
            rel_path = os.path.relpath(frame_path, os.path.join(self.dataset_root, 'real_test'))
            source = rel_path.split(os.sep)[0] if os.sep in rel_path else 'real_test'

            self.splits['test'].append({
                'image_path': frame_path,
                'label': 0,
                'label_name': 'real',
                'source': f'real_test_{source}',
                'video_id': Path(frame_path).stem
            })
            self.stats['test']['real'] += 1

        # Add real_test videos to test split
        real_test_videos = self.collect_real_test_videos()
        for frame_data in real_test_videos:
            self.splits['test'].append(frame_data)
            self.stats['test']['real'] += 1

        print(f"Train: {len(train_videos)} videos, "
              f"{sum(len(real_train_val[v]) for v in train_videos)} frames")
        print(f"Val: {len(val_videos)} videos, "
              f"{sum(len(real_train_val[v]) for v in val_videos)} frames")
        print(f"Test (images): {len(real_test)} frames (from real_test/ images)")
        print(f"Test (videos): {len(real_test_videos)} frames (from real_test/0_video/)")
        print(f"Test (total): {len(real_test) + len(real_test_videos)} real frames")

    def save_json_files(self):
        """Save train/val/test JSON files"""
        print("\n" + "=" * 80)
        print("SAVING JSON FILES")
        print("=" * 80)

        os.makedirs(self.output_dir, exist_ok=True)

        # Shuffle each split
        for split_name in ['train', 'val', 'test']:
            random.shuffle(self.splits[split_name])

        # Save each split
        for split_name in ['train', 'val', 'test']:
            output_path = os.path.join(self.output_dir, f'{split_name}.json')

            json_data = {
                'metadata': {
                    'split': split_name,
                    'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'total_samples': len(self.splits[split_name]),
                    'real_count': self.stats[split_name]['real'],
                    'fake_count': self.stats[split_name]['fake'],
                    'seed': self.seed
                },
                'data': self.splits[split_name]
            }

            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)

            print(f"{split_name}.json saved:")
            print(f"  Path: {output_path}")
            print(f"  Total: {len(self.splits[split_name])} samples")
            print(f"  Real: {self.stats[split_name]['real']}")
            print(f"  Fake: {self.stats[split_name]['fake']}")

    def print_summary(self):
        """Print final summary"""
        print("\n" + "=" * 80)
        print("DATASET JSON GENERATION SUMMARY")
        print("=" * 80)

        total_samples = sum(len(self.splits[s]) for s in ['train', 'val', 'test'])
        total_real = sum(self.stats[s]['real'] for s in ['train', 'val', 'test'])
        total_fake = sum(self.stats[s]['fake'] for s in ['train', 'val', 'test'])

        print(f"\nTRAIN:")
        print(f"  Total: {len(self.splits['train'])}")
        print(f"  Real: {self.stats['train']['real']}")
        print(f"  Fake: {self.stats['train']['fake']}")
        print(f"  Ratio: {self.stats['train']['real']}/{self.stats['train']['fake']} "
              f"({self.stats['train']['real']/self.stats['train']['fake']:.2f}:1)")

        print(f"\nVAL:")
        print(f"  Total: {len(self.splits['val'])}")
        print(f"  Real: {self.stats['val']['real']}")
        print(f"  Fake: {self.stats['val']['fake']}")
        print(f"  Ratio: {self.stats['val']['real']}/{self.stats['val']['fake']} "
              f"({self.stats['val']['real']/self.stats['val']['fake']:.2f}:1)")

        print(f"\nTEST:")
        print(f"  Total: {len(self.splits['test'])}")
        print(f"  Real: {self.stats['test']['real']}")
        print(f"  Fake: {self.stats['test']['fake']}")
        print(f"  Ratio: {self.stats['test']['real']}/{self.stats['test']['fake']} "
              f"({self.stats['test']['real']/self.stats['test']['fake']:.2f}:1)")

        print(f"\nTOTAL:")
        print(f"  Samples: {total_samples}")
        print(f"  Real: {total_real}")
        print(f"  Fake: {total_fake}")

        print(f"\n" + "=" * 80)

    def generate(self):
        """Run complete JSON generation"""
        print("\n" + "█" * 80)
        print("█" + " " * 78 + "█")
        print("█" + " " * 20 + "DATASET JSON GENERATOR" + " " * 37 + "█")
        print("█" + " " * 78 + "█")
        print("█" * 80)

        # Collect and split fake data
        fake_data = self.collect_fake_frames()
        self.split_fake_data(fake_data)

        # Collect and split real data
        real_train_val, real_test = self.collect_real_frames()
        self.split_real_data(real_train_val, real_test)

        # Save JSON files
        self.save_json_files()

        # Print summary
        self.print_summary()


if __name__ == '__main__':
    generator = DatasetJSONGenerator()
    generator.generate()
