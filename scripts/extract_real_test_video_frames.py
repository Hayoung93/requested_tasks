"""
Real Test Video Frame Extraction Script

This script extracts frames from videos in real_test/0_video/ directory.
These frames will be added to test.json for comprehensive evaluation.

Author: ForgeLens Team
Date: 2025-12-05
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import glob


class RealTestVideoFrameExtractor:
    def __init__(self,
                 video_root='/data/data/deepfake_finetune_dataset/real_test/0_video',
                 num_frames=8):
        """
        Initialize the Real Test Video Frame Extractor

        Args:
            video_root: Root directory containing real test videos
            num_frames: Number of frames to extract per video (default: 8)
        """
        self.video_root = video_root
        self.num_frames = num_frames
        self.stats = {
            'total_videos': 0,
            'total_frames': 0,
            'failed_videos': [],
            'sources': {}
        }

    def extract_frames_from_video(self, video_path, output_dir):
        """
        Extract frames from a single video using even interval sampling

        Args:
            video_path: Path to the video file
            output_dir: Directory to save extracted frames

        Returns:
            Number of frames successfully extracted
        """
        video_stem = Path(video_path).stem

        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                raise ValueError(f"Video has 0 frames: {video_path}")

            # Calculate frame indices using even interval sampling
            # Same logic as extract_frames.py
            if total_frames <= self.num_frames:
                # If video has fewer frames than requested, take all frames
                frame_indices = list(range(total_frames))
            else:
                # Use linspace for even interval sampling
                frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Extract and save frames
            extracted_count = 0
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()

                if ret:
                    # Save frame with format: {video_stem}_frame_{original_idx:06d}.png
                    frame_filename = f"{video_stem}_frame_{idx:06d}.png"
                    frame_path = os.path.join(output_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    extracted_count += 1

            cap.release()
            return extracted_count

        except Exception as e:
            print(f"\nError processing {video_path}: {str(e)}")
            self.stats['failed_videos'].append({
                'path': video_path,
                'error': str(e)
            })
            return 0

    def process_source_directory(self, source_dir):
        """
        Process all videos in a source directory

        Args:
            source_dir: Path to source directory containing videos

        Returns:
            Tuple of (video_count, frame_count)
        """
        source_name = os.path.basename(source_dir)

        # Find all video files
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI', '*.MOV']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(source_dir, ext)))

        if not video_files:
            print(f"\n  No videos found in {source_name}")
            return 0, 0

        print(f"\n{source_name}:")
        print(f"  Found {len(video_files)} videos")

        # Create frames output directory
        frames_dir = os.path.join(source_dir, 'frames')

        video_count = 0
        frame_count = 0

        # Process each video with progress bar
        for video_path in tqdm(video_files, desc=f"  Processing {source_name}",
                               unit="video", leave=False):
            extracted = self.extract_frames_from_video(video_path, frames_dir)
            if extracted > 0:
                video_count += 1
                frame_count += extracted

        print(f"  Videos processed: {video_count}")
        print(f"  Frames extracted: {frame_count}")

        return video_count, frame_count

    def process_all_videos(self):
        """
        Process all videos in real_test/0_video/0_real/
        """
        print("\n" + "=" * 80)
        print("EXTRACTING FRAMES FROM REAL TEST VIDEOS (0_video/)")
        print("=" * 80)
        print(f"Video root: {self.video_root}")
        print(f"Frames per video: {self.num_frames}")
        print(f"Sampling method: Even interval")

        # Find all source directories under 0_video/0_real/
        real_dir = os.path.join(self.video_root, '0_real')

        if not os.path.exists(real_dir):
            print(f"\nError: Directory not found: {real_dir}")
            return

        # Get all subdirectories in 0_real/
        source_dirs = [d for d in glob.glob(os.path.join(real_dir, '*'))
                      if os.path.isdir(d)]

        if not source_dirs:
            print(f"\nError: No source directories found in {real_dir}")
            return

        print(f"\nFound {len(source_dirs)} source directories:")
        for source_dir in source_dirs:
            print(f"  - {os.path.basename(source_dir)}")

        # Process each source directory
        for source_dir in source_dirs:
            source_name = os.path.basename(source_dir)
            video_count, frame_count = self.process_source_directory(source_dir)

            self.stats['sources'][source_name] = {
                'videos': video_count,
                'frames': frame_count
            }
            self.stats['total_videos'] += video_count
            self.stats['total_frames'] += frame_count

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print extraction summary"""
        print("\n" + "=" * 80)
        print("EXTRACTION COMPLETE")
        print("=" * 80)

        print("\nPer-source statistics:")
        for source, counts in self.stats['sources'].items():
            print(f"  {source}:")
            print(f"    Videos: {counts['videos']}")
            print(f"    Frames: {counts['frames']}")

        print(f"\nTotal statistics:")
        print(f"  Total videos processed: {self.stats['total_videos']}")
        print(f"  Total frames extracted: {self.stats['total_frames']}")
        print(f"  Expected frames: {self.stats['total_videos'] * self.num_frames}")
        print(f"  Failed videos: {len(self.stats['failed_videos'])}")

        if self.stats['failed_videos']:
            print("\nFailed videos:")
            for failed in self.stats['failed_videos']:
                print(f"  - {failed['path']}")
                print(f"    Error: {failed['error']}")

        print("\n" + "=" * 80)


def main():
    """Main execution function"""
    extractor = RealTestVideoFrameExtractor(
        video_root='/data/data/deepfake_finetune_dataset/real_test/0_video',
        num_frames=8
    )

    extractor.process_all_videos()


if __name__ == '__main__':
    main()
