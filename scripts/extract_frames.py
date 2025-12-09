"""
Frame Extraction Script for ForgeLens Custom Dataset

This script extracts frames from videos with even interval sampling:
1. sora_dataset/ videos: 8 frames per video
2. real/ videos: 13 frames per video (for 1:1 class balance)

Frames are saved as: {video_folder}/frames/{video_name}_frame_{original_idx}.png

Author: ForgeLens Team
Date: 2025-11-28
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


class FrameExtractor:
    def __init__(self, dataset_root='/data/data/deepfake_finetune_dataset'):
        self.dataset_root = dataset_root
        self.extraction_stats = {
            'sora': {'total': 0, 'success': 0, 'failed': 0, 'warnings': []},
            'real': {'total': 0, 'success': 0, 'failed': 0, 'warnings': []}
        }

    def extract_frames_even_interval(self, video_path, output_dir, num_frames, video_name):
        """
        Extract frames at even intervals from video

        Args:
            video_path: Path to video file
            output_dir: Output directory for frames
            num_frames: Number of frames to extract
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

        # Calculate even interval indices
        if num_frames == 1:
            indices = [total_frames // 2]  # Middle frame
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Extract frames
        extracted_count = 0
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                frame_filename = f"{video_name}_frame_{idx:06d}.png"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                extracted_count += 1
            else:
                warning_msg = f"Failed to read frame {idx} from {video_path}"

        cap.release()

        success = extracted_count > 0
        return success, extracted_count, warning_msg

    def extract_sora_frames(self, frames_per_video=8):
        """Extract frames from sora_dataset/ videos"""
        print("\n" + "=" * 80)
        print("EXTRACTING FRAMES FROM SORA DATASET")
        print("=" * 80)
        print(f"Frames per video: {frames_per_video}")

        sora_dir = os.path.join(self.dataset_root, 'sora_dataset')
        if not os.path.exists(sora_dir):
            print(f"ERROR: {sora_dir} not found")
            return

        # Find all MP4 videos
        video_files = glob.glob(os.path.join(sora_dir, '*.mp4'))
        self.extraction_stats['sora']['total'] = len(video_files)

        print(f"Found {len(video_files)} videos")
        print("Extracting frames...")

        # Create output directory
        output_base_dir = os.path.join(sora_dir, 'frames')

        # Extract frames from each video
        for video_path in tqdm(video_files, desc="Sora videos"):
            video_name = Path(video_path).stem  # Get filename without extension

            # Extract frames
            success, extracted, warning = self.extract_frames_even_interval(
                video_path=video_path,
                output_dir=output_base_dir,
                num_frames=frames_per_video,
                video_name=video_name
            )

            if success:
                self.extraction_stats['sora']['success'] += 1
            else:
                self.extraction_stats['sora']['failed'] += 1

            if warning:
                self.extraction_stats['sora']['warnings'].append({
                    'video': video_name,
                    'message': warning
                })

        # Print stats
        print(f"\nSora Extraction Complete:")
        print(f"  - Total videos: {self.extraction_stats['sora']['total']}")
        print(f"  - Successful: {self.extraction_stats['sora']['success']}")
        print(f"  - Failed: {self.extraction_stats['sora']['failed']}")
        print(f"  - Warnings: {len(self.extraction_stats['sora']['warnings'])}")

        if self.extraction_stats['sora']['warnings']:
            print(f"\n  Warnings:")
            for w in self.extraction_stats['sora']['warnings'][:10]:  # Show first 10
                print(f"    - {w['video']}: {w['message']}")
            if len(self.extraction_stats['sora']['warnings']) > 10:
                print(f"    ... and {len(self.extraction_stats['sora']['warnings']) - 10} more")

    def extract_real_frames(self, frames_per_video=13):
        """Extract frames from real/ videos"""
        print("\n" + "=" * 80)
        print("EXTRACTING FRAMES FROM REAL DATASET")
        print("=" * 80)
        print(f"Frames per video: {frames_per_video}")

        real_dir = os.path.join(self.dataset_root, 'real', '1. klleon_real_dataset')
        if not os.path.exists(real_dir):
            print(f"ERROR: {real_dir} not found")
            return

        # Find all MP4 videos recursively
        video_files = glob.glob(os.path.join(real_dir, '**', '*.mp4'), recursive=True)
        self.extraction_stats['real']['total'] = len(video_files)

        print(f"Found {len(video_files)} videos")
        print("Extracting frames...")

        # Extract frames from each video
        for video_path in tqdm(video_files, desc="Real videos"):
            video_name = Path(video_path).stem
            video_dir = Path(video_path).parent

            # Output directory: same directory as video, in frames/ subfolder
            output_dir = os.path.join(video_dir, 'frames')

            # Extract frames
            success, extracted, warning = self.extract_frames_even_interval(
                video_path=video_path,
                output_dir=output_dir,
                num_frames=frames_per_video,
                video_name=video_name
            )

            if success:
                self.extraction_stats['real']['success'] += 1
            else:
                self.extraction_stats['real']['failed'] += 1

            if warning:
                self.extraction_stats['real']['warnings'].append({
                    'video': video_name,
                    'path': video_path,
                    'message': warning
                })

        # Print stats
        print(f"\nReal Extraction Complete:")
        print(f"  - Total videos: {self.extraction_stats['real']['total']}")
        print(f"  - Successful: {self.extraction_stats['real']['success']}")
        print(f"  - Failed: {self.extraction_stats['real']['failed']}")
        print(f"  - Warnings: {len(self.extraction_stats['real']['warnings'])}")

        if self.extraction_stats['real']['warnings']:
            print(f"\n  Warnings:")
            for w in self.extraction_stats['real']['warnings'][:10]:  # Show first 10
                print(f"    - {w['video']}: {w['message']}")
            if len(self.extraction_stats['real']['warnings']) > 10:
                print(f"    ... and {len(self.extraction_stats['real']['warnings']) - 10} more")

    def save_extraction_report(self):
        """Save extraction report to file"""
        report_path = '/workspace/ForgeLens/scripts/frame_extraction_report.txt'

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FRAME EXTRACTION REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Sora stats
            f.write("SORA DATASET:\n")
            f.write(f"  - Total videos: {self.extraction_stats['sora']['total']}\n")
            f.write(f"  - Successful: {self.extraction_stats['sora']['success']}\n")
            f.write(f"  - Failed: {self.extraction_stats['sora']['failed']}\n")
            f.write(f"  - Expected frames: {self.extraction_stats['sora']['success'] * 8}\n")
            f.write(f"  - Warnings: {len(self.extraction_stats['sora']['warnings'])}\n\n")

            if self.extraction_stats['sora']['warnings']:
                f.write("  Sora Warnings:\n")
                for w in self.extraction_stats['sora']['warnings']:
                    f.write(f"    - {w['video']}: {w['message']}\n")
                f.write("\n")

            # Real stats
            f.write("REAL DATASET:\n")
            f.write(f"  - Total videos: {self.extraction_stats['real']['total']}\n")
            f.write(f"  - Successful: {self.extraction_stats['real']['success']}\n")
            f.write(f"  - Failed: {self.extraction_stats['real']['failed']}\n")
            f.write(f"  - Expected frames: {self.extraction_stats['real']['success'] * 13}\n")
            f.write(f"  - Warnings: {len(self.extraction_stats['real']['warnings'])}\n\n")

            if self.extraction_stats['real']['warnings']:
                f.write("  Real Warnings:\n")
                for w in self.extraction_stats['real']['warnings']:
                    f.write(f"    - {w['video']}: {w['message']}\n")
                    f.write(f"      Path: {w['path']}\n")
                f.write("\n")

            # Total
            total_videos = self.extraction_stats['sora']['total'] + self.extraction_stats['real']['total']
            total_success = self.extraction_stats['sora']['success'] + self.extraction_stats['real']['success']
            total_expected_frames = (self.extraction_stats['sora']['success'] * 8 +
                                    self.extraction_stats['real']['success'] * 13)

            f.write("TOTAL:\n")
            f.write(f"  - Total videos processed: {total_videos}\n")
            f.write(f"  - Successful extractions: {total_success}\n")
            f.write(f"  - Expected total frames: {total_expected_frames}\n")

        print(f"\nExtraction report saved to: {report_path}")

    def run_extraction(self):
        """Run complete frame extraction"""
        print("\n" + "█" * 80)
        print("█" + " " * 78 + "█")
        print("█" + " " * 22 + "FRAME EXTRACTOR" + " " * 41 + "█")
        print("█" + " " * 78 + "█")
        print("█" * 80)

        # Extract sora frames (8 per video)
        self.extract_sora_frames(frames_per_video=8)

        # Extract real frames (13 per video for 1:1 balance)
        self.extract_real_frames(frames_per_video=13)

        # Save report
        self.save_extraction_report()

        print("\n" + "=" * 80)
        print("EXTRACTION COMPLETE")
        print("=" * 80)


if __name__ == '__main__':
    extractor = FrameExtractor()
    extractor.run_extraction()
