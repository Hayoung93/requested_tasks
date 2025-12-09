"""
Dataset Analysis Script for ForgeLens Custom Dataset

This script analyzes the custom deepfake dataset and calculates:
1. Number of fake images/videos across all sources
2. Number of real videos
3. Required frames per real video for 1:1 class balance

Author: ForgeLens Team
Date: 2025-11-28
"""

import os
import glob
from pathlib import Path
import cv2


class DatasetAnalyzer:
    def __init__(self, dataset_root='/data/data/deepfake_finetune_dataset'):
        self.dataset_root = dataset_root
        self.stats = {
            'fake': {},
            'real': {},
            'summary': {}
        }

    def count_frames_in_directory(self, directory, pattern='*.png'):
        """Count image files in a directory"""
        files = glob.glob(os.path.join(directory, '**', pattern), recursive=True)
        return len(files)

    def count_videos_in_directory(self, directory, pattern='*.mp4'):
        """Count video files in a directory"""
        files = glob.glob(os.path.join(directory, '**', pattern), recursive=True)
        return len(files), files

    def get_video_info(self, video_path):
        """Get video information (frame count, fps, duration)"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0

        cap.release()
        return {
            'frame_count': frame_count,
            'fps': fps,
            'duration': duration
        }

    def analyze_fake_class(self):
        """Analyze all fake/AI-generated image sources"""
        print("\n" + "=" * 80)
        print("ANALYZING FAKE CLASS")
        print("=" * 80)

        # 1. kling2.5_dataset (extracted frames)
        kling_dir = os.path.join(self.dataset_root, 'kling2.5_dataset')
        if os.path.exists(kling_dir):
            frame_count = self.count_frames_in_directory(kling_dir)
            video_dirs = [d for d in glob.glob(os.path.join(kling_dir, '*')) if os.path.isdir(d)]
            self.stats['fake']['kling2.5'] = {
                'frames': frame_count,
                'videos': len(video_dirs),
                'type': 'extracted_frames'
            }
            print(f"\n1. kling2.5_dataset:")
            print(f"   - Frames: {frame_count:,}")
            print(f"   - Video directories: {len(video_dirs)}")
        else:
            print(f"\nWARNING: {kling_dir} not found")

        # 2. veo3_dataset (extracted frames)
        veo_dir = os.path.join(self.dataset_root, 'veo3_dataset')
        if os.path.exists(veo_dir):
            frame_count = self.count_frames_in_directory(veo_dir)
            video_dirs = [d for d in glob.glob(os.path.join(veo_dir, '*')) if os.path.isdir(d)]
            self.stats['fake']['veo3'] = {
                'frames': frame_count,
                'videos': len(video_dirs),
                'type': 'extracted_frames'
            }
            print(f"\n2. veo3_dataset:")
            print(f"   - Frames: {frame_count:,}")
            print(f"   - Video directories: {len(video_dirs)}")
        else:
            print(f"\nWARNING: {veo_dir} not found")

        # 3. nanobanana_dataset (flat images)
        nano_dir = os.path.join(self.dataset_root, 'nanobanana_dataset')
        if os.path.exists(nano_dir):
            frame_count = self.count_frames_in_directory(nano_dir)
            self.stats['fake']['nanobanana'] = {
                'frames': frame_count,
                'videos': frame_count,  # Each image is treated as separate video
                'type': 'flat_images'
            }
            print(f"\n3. nanobanana_dataset:")
            print(f"   - Images: {frame_count:,}")
        else:
            print(f"\nWARNING: {nano_dir} not found")

        # 4. sora_dataset (videos to be extracted)
        sora_dir = os.path.join(self.dataset_root, 'sora_dataset')
        if os.path.exists(sora_dir):
            video_count, video_files = self.count_videos_in_directory(sora_dir)
            expected_frames = video_count * 8  # 8 frames per video

            # Analyze a sample of videos
            sample_videos = video_files[:5]  # Check first 5 videos
            video_info_list = []
            for video_path in sample_videos:
                info = self.get_video_info(video_path)
                if info:
                    video_info_list.append(info)

            avg_duration = sum(v['duration'] for v in video_info_list) / len(video_info_list) if video_info_list else 0
            min_frames = min(v['frame_count'] for v in video_info_list) if video_info_list else 0
            max_frames = max(v['frame_count'] for v in video_info_list) if video_info_list else 0

            self.stats['fake']['sora'] = {
                'videos': video_count,
                'expected_frames': expected_frames,
                'type': 'videos',
                'avg_duration': avg_duration,
                'frame_range': (min_frames, max_frames)
            }
            print(f"\n4. sora_dataset:")
            print(f"   - Videos: {video_count:,}")
            print(f"   - Expected frames (8/video): {expected_frames:,}")
            print(f"   - Sample video info (first 5):")
            print(f"     - Avg duration: {avg_duration:.2f}s")
            print(f"     - Frame count range: {min_frames} - {max_frames}")
        else:
            print(f"\nWARNING: {sora_dir} not found")

        # Calculate total fake frames
        total_fake = 0
        total_fake += self.stats['fake'].get('kling2.5', {}).get('frames', 0)
        total_fake += self.stats['fake'].get('veo3', {}).get('frames', 0)
        total_fake += self.stats['fake'].get('nanobanana', {}).get('frames', 0)
        total_fake += self.stats['fake'].get('sora', {}).get('expected_frames', 0)

        self.stats['summary']['total_fake_frames'] = total_fake

        print(f"\n{'─' * 80}")
        print(f"TOTAL FAKE FRAMES: {total_fake:,}")
        print(f"{'─' * 80}")

    def analyze_real_class(self):
        """Analyze real video sources"""
        print("\n" + "=" * 80)
        print("ANALYZING REAL CLASS")
        print("=" * 80)

        # 1. real/ folder (videos to be extracted for train/val)
        real_dir = os.path.join(self.dataset_root, 'real', '1. klleon_real_dataset')
        if os.path.exists(real_dir):
            video_count, video_files = self.count_videos_in_directory(real_dir)

            # Analyze sample videos
            sample_videos = video_files[:10]  # Check first 10 videos
            video_info_list = []
            for video_path in sample_videos:
                info = self.get_video_info(video_path)
                if info:
                    video_info_list.append(info)

            if video_info_list:
                avg_frame_count = sum(v['frame_count'] for v in video_info_list) / len(video_info_list)
                avg_duration = sum(v['duration'] for v in video_info_list) / len(video_info_list)
                min_frames = min(v['frame_count'] for v in video_info_list)
                max_frames = max(v['frame_count'] for v in video_info_list)
            else:
                avg_frame_count = 0
                avg_duration = 0
                min_frames = 0
                max_frames = 0

            self.stats['real']['videos'] = {
                'count': video_count,
                'avg_frame_count': avg_frame_count,
                'avg_duration': avg_duration,
                'frame_range': (min_frames, max_frames)
            }

            print(f"\n1. real/1. klleon_real_dataset:")
            print(f"   - Videos: {video_count:,}")
            print(f"   - Sample video info (first 10):")
            print(f"     - Avg duration: {avg_duration:.2f}s")
            print(f"     - Avg frame count: {avg_frame_count:.1f}")
            print(f"     - Frame count range: {min_frames} - {max_frames}")

            # Calculate required frames per video for 1:1 balance
            # We need to match total_fake_frames with real frames for train/val
            total_fake = self.stats['summary']['total_fake_frames']

            # Split: train/val/test = 8:1:1 for fake
            fake_train = int(total_fake * 0.8)
            fake_val = int(total_fake * 0.1)

            # For real: train/val = 8:2 (test comes from real_test folder)
            # So train+val should equal fake_train+fake_val
            real_train_val_needed = fake_train + fake_val

            # Number of real videos for train/val
            real_videos_for_train_val = video_count  # All videos used for train/val

            # Frames per video needed
            frames_per_video = int(real_train_val_needed / real_videos_for_train_val) + 1

            self.stats['real']['frames_per_video_needed'] = frames_per_video
            self.stats['summary']['real_train_val_frames_needed'] = real_train_val_needed

            print(f"\n   {'─' * 76}")
            print(f"   Balance Calculation:")
            print(f"   - Fake train+val frames: {fake_train + fake_val:,}")
            print(f"   - Real videos available: {video_count:,}")
            print(f"   - FRAMES PER REAL VIDEO NEEDED: {frames_per_video}")
            print(f"   - Total real frames (estimate): {frames_per_video * video_count:,}")
            print(f"   {'─' * 76}")

        else:
            print(f"\nWARNING: {real_dir} not found")

        # 2. real_test/ folder (for test set)
        real_test_dir = os.path.join(self.dataset_root, 'real_test')
        if os.path.exists(real_test_dir):
            frame_count = self.count_frames_in_directory(real_test_dir)
            self.stats['real']['test_frames'] = frame_count

            print(f"\n2. real_test/:")
            print(f"   - Images (for test set): {frame_count:,}")
        else:
            print(f"\nWARNING: {real_test_dir} not found")

    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "=" * 80)
        print("DATASET SUMMARY")
        print("=" * 80)

        total_fake = self.stats['summary'].get('total_fake_frames', 0)
        frames_per_video = self.stats['real'].get('frames_per_video_needed', 0)
        real_videos = self.stats['real'].get('videos', {}).get('count', 0)
        real_train_val = frames_per_video * real_videos
        real_test = self.stats['real'].get('test_frames', 0)

        # Split calculations
        fake_train = int(total_fake * 0.8)
        fake_val = int(total_fake * 0.1)
        fake_test = total_fake - fake_train - fake_val

        real_train = int(real_train_val * 0.8)
        real_val = real_train_val - real_train

        print(f"\nFAKE CLASS:")
        print(f"  - Total frames: {total_fake:,}")
        print(f"  - Train (80%): {fake_train:,}")
        print(f"  - Val (10%): {fake_val:,}")
        print(f"  - Test (10%): {fake_test:,}")

        print(f"\nREAL CLASS:")
        print(f"  - Videos for train/val: {real_videos:,}")
        print(f"  - Frames per video: {frames_per_video}")
        print(f"  - Train+Val frames: {real_train_val:,}")
        print(f"    - Train (80%): {real_train:,}")
        print(f"    - Val (20%): {real_val:,}")
        print(f"  - Test frames (real_test/): {real_test:,}")

        print(f"\nTOTAL DATASET:")
        print(f"  - Train: {fake_train + real_train:,} ({real_train:,} real + {fake_train:,} fake)")
        print(f"  - Val: {fake_val + real_val:,} ({real_val:,} real + {fake_val:,} fake)")
        print(f"  - Test: {fake_test + real_test:,} ({real_test:,} real + {fake_test:,} fake)")
        print(f"  - TOTAL: {total_fake + real_train_val + real_test:,}")

        print(f"\n{'=' * 80}\n")

        # Save to file
        summary_path = '/workspace/ForgeLens/scripts/dataset_analysis_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DATASET ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write("FAKE CLASS:\n")
            f.write(f"  - Total frames: {total_fake:,}\n")
            f.write(f"  - Train (80%): {fake_train:,}\n")
            f.write(f"  - Val (10%): {fake_val:,}\n")
            f.write(f"  - Test (10%): {fake_test:,}\n\n")

            f.write("REAL CLASS:\n")
            f.write(f"  - Videos for train/val: {real_videos:,}\n")
            f.write(f"  - Frames per video: {frames_per_video}\n")
            f.write(f"  - Train+Val frames: {real_train_val:,}\n")
            f.write(f"    - Train (80%): {real_train:,}\n")
            f.write(f"    - Val (20%): {real_val:,}\n")
            f.write(f"  - Test frames (real_test/): {real_test:,}\n\n")

            f.write("TOTAL DATASET:\n")
            f.write(f"  - Train: {fake_train + real_train:,} ({real_train:,} real + {fake_train:,} fake)\n")
            f.write(f"  - Val: {fake_val + real_val:,} ({real_val:,} real + {fake_val:,} fake)\n")
            f.write(f"  - Test: {fake_test + real_test:,} ({real_test:,} real + {fake_test:,} fake)\n")
            f.write(f"  - TOTAL: {total_fake + real_train_val + real_test:,}\n")

        print(f"Summary saved to: {summary_path}")

        return self.stats

    def run_analysis(self):
        """Run complete dataset analysis"""
        print("\n" + "█" * 80)
        print("█" + " " * 78 + "█")
        print("█" + " " * 20 + "FORGELENS DATASET ANALYZER" + " " * 32 + "█")
        print("█" + " " * 78 + "█")
        print("█" * 80)

        self.analyze_fake_class()
        self.analyze_real_class()
        stats = self.print_summary()

        return stats


if __name__ == '__main__':
    analyzer = DatasetAnalyzer()
    stats = analyzer.run_analysis()
