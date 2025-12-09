"""
Generate test_nfs.json for NFS_data

This script:
1. Extracts frames from NFS_data videos (8 frames per video, even interval)
2. Generates test_nfs.json with all extracted frames

NFS_data structure:
- 10 fake generation methods (ComfyUI, DeepbrainAI, DeevidAI, HailouAI, ImagineArt,
  KlingAI, LumaLabs, Sora, Veo3, WanAI)
- 1 real folder (Real)

Author: ForgeLens Team
Date: 2025-12-02
"""

import os
import json
import glob
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime


class NFSDataProcessor:
    def __init__(self, nfs_root='/data/data/deepfake_finetune_dataset/NFS_data',
                 output_json='/data/data/deepfake_finetune_dataset/jsons/test_nfs.json',
                 frames_per_video=8):
        self.nfs_root = nfs_root
        self.output_json = output_json
        self.frames_per_video = frames_per_video

        self.data = []
        self.stats = {'real': 0, 'fake': 0, 'sources': {}}

        # Fake generation methods
        self.fake_sources = [
            'ComfyUI', 'DeepbrainAI', 'DeevidAI', 'HailouAI', 'ImagineArt',
            'KlingAI', 'LumaLabs', 'Sora', 'Veo3', 'WanAI'
        ]

        # Real folder
        self.real_source = 'Real'

    def extract_frames_even_interval(self, video_path, output_dir, num_frames, video_name):
        """Extract frames at even intervals from video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, 0, f"Failed to open video: {video_path}"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check if video is too short
        warning_msg = None
        if total_frames < num_frames:
            warning_msg = f"Video has only {total_frames} frames (requested {num_frames})"
            num_frames = total_frames

        # Calculate even interval indices
        if num_frames == 1:
            indices = [total_frames // 2]
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Extract frames
        extracted_count = 0
        frame_paths = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                frame_filename = f"{video_name}_frame_{idx:06d}.png"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                extracted_count += 1

        cap.release()

        success = extracted_count > 0
        return success, frame_paths, warning_msg

    def process_fake_sources(self):
        """Process all fake generation method folders"""
        print("\n" + "=" * 80)
        print("PROCESSING FAKE SOURCES")
        print("=" * 80)

        for source in self.fake_sources:
            source_dir = os.path.join(self.nfs_root, source)

            if not os.path.exists(source_dir):
                print(f"\nWARNING: {source} folder not found, skipping...")
                continue

            print(f"\nProcessing {source}...")

            # Find all video files
            video_files = []
            video_files.extend(glob.glob(os.path.join(source_dir, '*.mp4')))
            video_files.extend(glob.glob(os.path.join(source_dir, '*.avi')))
            video_files.extend(glob.glob(os.path.join(source_dir, '*.mov')))
            video_files.extend(glob.glob(os.path.join(source_dir, '**', '*.mp4'), recursive=True))
            video_files.extend(glob.glob(os.path.join(source_dir, '**', '*.avi'), recursive=True))
            video_files.extend(glob.glob(os.path.join(source_dir, '**', '*.mov'), recursive=True))

            # Remove duplicates
            video_files = list(set(video_files))

            print(f"  Found {len(video_files)} videos")

            # Create frames directory
            frames_dir = os.path.join(source_dir, 'frames')

            # Process videos
            source_frame_count = 0
            for video_path in tqdm(video_files, desc=f"  {source}"):
                video_name = Path(video_path).stem

                # Extract frames
                success, frame_paths, warning = self.extract_frames_even_interval(
                    video_path=video_path,
                    output_dir=frames_dir,
                    num_frames=self.frames_per_video,
                    video_name=video_name
                )

                if success and frame_paths:
                    # Add to data
                    for frame_path in frame_paths:
                        self.data.append({
                            'image_path': frame_path,
                            'label': 1,
                            'label_name': 'fake',
                            'source': source,
                            'video_id': video_name
                        })
                        source_frame_count += 1
                        self.stats['fake'] += 1

            self.stats['sources'][source] = source_frame_count
            print(f"  ✓ Extracted {source_frame_count} frames from {source}")

    def process_real_source(self):
        """Process Real folder"""
        print("\n" + "=" * 80)
        print("PROCESSING REAL SOURCE")
        print("=" * 80)

        real_dir = os.path.join(self.nfs_root, self.real_source)

        if not os.path.exists(real_dir):
            print(f"ERROR: {real_dir} not found")
            return

        print(f"\nProcessing Real...")

        # Find all video files
        video_files = []
        video_files.extend(glob.glob(os.path.join(real_dir, '*.mp4')))
        video_files.extend(glob.glob(os.path.join(real_dir, '*.avi')))
        video_files.extend(glob.glob(os.path.join(real_dir, '*.mov')))
        video_files.extend(glob.glob(os.path.join(real_dir, '**', '*.mp4'), recursive=True))
        video_files.extend(glob.glob(os.path.join(real_dir, '**', '*.avi'), recursive=True))
        video_files.extend(glob.glob(os.path.join(real_dir, '**', '*.mov'), recursive=True))

        # Remove duplicates
        video_files = list(set(video_files))

        print(f"  Found {len(video_files)} videos")

        # Create frames directory
        frames_dir = os.path.join(real_dir, 'frames')

        # Process videos
        real_frame_count = 0
        for video_path in tqdm(video_files, desc="  Real"):
            video_name = Path(video_path).stem

            # Extract frames
            success, frame_paths, warning = self.extract_frames_even_interval(
                video_path=video_path,
                output_dir=frames_dir,
                num_frames=self.frames_per_video,
                video_name=video_name
            )

            if success and frame_paths:
                # Add to data
                for frame_path in frame_paths:
                    self.data.append({
                        'image_path': frame_path,
                        'label': 0,
                        'label_name': 'real',
                        'source': 'Real',
                        'video_id': video_name
                    })
                    real_frame_count += 1
                    self.stats['real'] += 1

        self.stats['sources']['Real'] = real_frame_count
        print(f"  ✓ Extracted {real_frame_count} frames from Real")

    def save_json(self):
        """Save test_nfs.json"""
        print("\n" + "=" * 80)
        print("SAVING test_nfs.json")
        print("=" * 80)

        # Create metadata
        metadata = {
            'split': 'test_nfs',
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': len(self.data),
            'real_count': self.stats['real'],
            'fake_count': self.stats['fake'],
            'frames_per_video': self.frames_per_video,
            'sources': self.stats['sources']
        }

        # Create JSON structure
        json_data = {
            'metadata': metadata,
            'data': self.data
        }

        # Save to file
        os.makedirs(os.path.dirname(self.output_json), exist_ok=True)
        with open(self.output_json, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"\ntest_nfs.json saved:")
        print(f"  Path: {self.output_json}")
        print(f"  Total samples: {len(self.data)}")
        print(f"  Real: {self.stats['real']}")
        print(f"  Fake: {self.stats['fake']}")
        print(f"\nSource breakdown:")
        for source, count in sorted(self.stats['sources'].items()):
            label = 'real' if source == 'Real' else 'fake'
            print(f"  {source} ({label}): {count}")

    def generate(self):
        """Run complete NFS test JSON generation"""
        print("\n" + "█" * 80)
        print("█" + " " * 78 + "█")
        print("█" + " " * 20 + "NFS TEST JSON GENERATOR" + " " * 35 + "█")
        print("█" + " " * 78 + "█")
        print("█" * 80)

        # Process fake sources
        self.process_fake_sources()

        # Process real source
        self.process_real_source()

        # Save JSON
        self.save_json()

        print("\n" + "=" * 80)
        print("GENERATION COMPLETE")
        print("=" * 80)


if __name__ == '__main__':
    processor = NFSDataProcessor()
    processor.generate()
