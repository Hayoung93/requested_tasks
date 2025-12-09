#!/usr/bin/env python3
"""
Extract frames from youtube_real videos using the same method as klleon_real

This script extracts 13 frames per video using even interval sampling,
matching the klleon_real dataset preparation.

Usage:
    python scripts/extract_youtube_real_frames.py

Output:
    Frames saved to: /data/data/deepfake_finetune_dataset/real/youtube_real/frames/
    Naming format: {video_name}_frame_{idx:06d}.png
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


class YouTubeRealFrameExtractor:
    def __init__(self, dataset_root='/data/data/deepfake_finetune_dataset'):
        self.dataset_root = dataset_root
        self.youtube_real_dir = os.path.join(dataset_root, 'real', 'youtube_real')
        self.output_dir = os.path.join(self.youtube_real_dir, 'frames')

        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'warnings': []
        }

    def extract_frames_even_interval(self, video_path, num_frames, video_name):
        """
        Extract frames at even intervals from video (same as klleon_real)

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract (13)
            video_name: Base name for output files

        Returns:
            tuple: (success, num_extracted, warning_message)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, 0, f"Failed to open video: {video_path}"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check if video is too short
        warning_msg = None
        if total_frames < num_frames:
            warning_msg = f"Video has only {total_frames} frames (requested {num_frames})"
            num_frames = total_frames  # Extract available frames

        # Calculate even interval indices (same as klleon_real)
        if num_frames == 1:
            indices = [total_frames // 2]  # Middle frame
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        # Extract frames
        extracted_count = 0
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                # Match klleon_real naming: {video_name}_frame_{idx:06d}.png
                frame_filename = f"{video_name}_frame_{idx:06d}.png"
                frame_path = os.path.join(self.output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                extracted_count += 1
            else:
                if warning_msg is None:
                    warning_msg = f"Failed to read frame {idx} from {video_path}"

        cap.release()

        success = extracted_count > 0
        return success, extracted_count, warning_msg

    def extract_all_videos(self, frames_per_video=13):
        """
        Extract frames from all youtube_real videos

        Args:
            frames_per_video: Number of frames to extract per video (default: 13)
        """
        print("=" * 70)
        print("YouTube Real Frame Extraction")
        print("=" * 70)
        print(f"Source directory: {self.youtube_real_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Frames per video: {frames_per_video}")
        print("=" * 70)
        print()

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Find all MP4 videos in youtube_real
        video_files = glob.glob(os.path.join(self.youtube_real_dir, '*.mp4'))
        video_files = sorted(video_files)  # Sort for reproducibility

        if len(video_files) == 0:
            print(f"ERROR: No MP4 files found in {self.youtube_real_dir}")
            return

        print(f"Found {len(video_files)} videos")
        print()

        # Extract frames from each video
        for video_path in tqdm(video_files, desc="Extracting frames"):
            self.stats['total'] += 1
            video_name = Path(video_path).stem  # Get filename without extension

            success, extracted, warning = self.extract_frames_even_interval(
                video_path=video_path,
                num_frames=frames_per_video,
                video_name=video_name
            )

            if success:
                self.stats['success'] += 1
            else:
                self.stats['failed'] += 1

            if warning:
                self.stats['warnings'].append(f"{video_name}: {warning}")

        # Print summary
        print()
        print("=" * 70)
        print("Extraction Complete!")
        print("=" * 70)
        print(f"Total videos processed: {self.stats['total']}")
        print(f"Successfully extracted: {self.stats['success']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Warnings: {len(self.stats['warnings'])}")
        print()

        if self.stats['warnings']:
            print("Warnings:")
            for warning in self.stats['warnings'][:10]:  # Show first 10
                print(f"  - {warning}")
            if len(self.stats['warnings']) > 10:
                print(f"  ... and {len(self.stats['warnings']) - 10} more")
            print()

        # Count extracted frames
        extracted_frames = len(glob.glob(os.path.join(self.output_dir, '*.png')))
        expected_frames = self.stats['success'] * frames_per_video
        print(f"Extracted frames: {extracted_frames}")
        print(f"Expected frames: {expected_frames}")

        if extracted_frames == expected_frames:
            print("✓ Frame count matches expected!")
        else:
            print(f"⚠ Frame count mismatch (difference: {abs(extracted_frames - expected_frames)})")

        print("=" * 70)


if __name__ == '__main__':
    extractor = YouTubeRealFrameExtractor()
    extractor.extract_all_videos(frames_per_video=13)
