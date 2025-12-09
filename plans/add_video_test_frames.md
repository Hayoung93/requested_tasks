# Implementation Plan: Add real_test/0_video/ Videos to test.json

**Date**: 2025-12-05
**Author**: ForgeLens Team
**Status**: Planned

---

## 1. Current State Analysis

### What We Have
- **Current test.json**: 1,483 samples (Real: 369, Fake: 1,114)
- **Real test sources currently included**:
  - `real_test_0_real`: 246 images (from real_test/0_real/1_image/)
  - `real_test_06_car_model`: 71 images
  - `real_test_2_뷰티픽`: 52 images
- **Total real samples**: 369 (all from 1_image folders)

### What's Missing
- **Location**: `/data/data/deepfake_finetune_dataset/real_test/0_video/`
- **Structure**:
  ```
  0_video/
  └── 0_real/
      ├── real_20250326_Youtube/
      ├── real_outputs_20250408_refine/
      └── FF++C23/
  ```
- **Total videos**: 211 .mp4 files
- **Expected frames**: 211 videos × 8 frames = 1,688 additional frames
- **These are real videos that should be included in test evaluation**

---

## 2. Problem Statement

The current test.json only includes image files from `real_test/` but excludes videos from `real_test/0_video/`. This means we're missing 211 real videos from our test set, which could affect:
- **Test set balance**: Real samples will increase from 369 to 2,057 (369 + 1,688)
- **Evaluation coverage**: Missing evaluation on video-based real samples
- **Final test.json**: Total samples will be ~3,171 (Real: 2,057, Fake: 1,114)

---

## 3. Solution Strategy

### Frame Extraction Approach
1. **Use even interval sampling** (same as existing implementation)
2. **Extract 8 frames per video**
3. **Save frames** to: `/data/data/deepfake_finetune_dataset/real_test/0_video/0_real/{source}/frames/`
4. **Naming convention**: `{video_stem}_frame_{original_idx:06d}.png`
   - Example: `009222_frame_000015.png`

### JSON Update Approach
- **Modify existing `generate_dataset_json.py`**
- Add new method: `collect_real_test_videos()`
- Add frames to test split with metadata:
  ```json
  {
    "image_path": "/data/.../real_test/0_video/0_real/real_20250326_Youtube/frames/009222_frame_000015.png",
    "label": 0,
    "label_name": "real",
    "source": "real_test_0_video_real_20250326_Youtube",
    "video_id": "009222"
  }
  ```

---

## 4. Implementation Steps

### Step 1: Create Frame Extraction Script
**File**: `/workspace/ForgeLens/scripts/extract_real_test_video_frames.py`

**Key components**:
```python
class RealTestVideoFrameExtractor:
    def __init__(self, video_root='/data/data/deepfake_finetune_dataset/real_test/0_video',
                 num_frames=8):
        self.video_root = video_root
        self.num_frames = num_frames

    def extract_frames_from_video(self, video_path, output_dir):
        """Extract 8 frames using even interval sampling"""
        # Same logic as extract_frames.py
        # Use cv2.VideoCapture
        # Calculate indices with np.linspace
        # Save as {video_stem}_frame_{idx:06d}.png

    def process_all_videos(self):
        """Process all videos in 0_video/0_real/"""
        # Find all .mp4, .avi, .mov files
        # Create frames/ subdirectories
        # Extract frames for each video
```

**Expected output**:
- Frames saved to: `real_test/0_video/0_real/{source}/frames/`
- Total frames: 211 videos × 8 = 1,688 frames

### Step 2: Modify generate_dataset_json.py
**File**: `/workspace/ForgeLens/scripts/generate_dataset_json.py`

**Add new method**:
```python
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
                video_id = '_'.join(frame_name.split('_')[:-2])  # Remove '_frame_{idx}'

                real_test_video_frames.append({
                    'image_path': frame_path,
                    'label': 0,
                    'label_name': 'real',
                    'source': f'real_test_0_video_{source}',
                    'video_id': video_id
                })

        print(f"real_test/0_video/: {len(real_test_video_frames)} frames")

    return real_test_video_frames
```

**Modify split_real_data() method**:
```python
def split_real_data(self, real_train_val, real_test):
    """Split real data 8:2 for train/val, use real_test for test"""
    # ... existing code for train/val split ...

    # Add real_test images (existing logic)
    for frame_path in real_test:
        # ... existing code ...

    # NEW: Add real_test videos
    real_test_videos = self.collect_real_test_videos()
    for frame_data in real_test_videos:
        self.splits['test'].append(frame_data)
        self.stats['test']['real'] += 1

    print(f"Test (0_video/): {len(real_test_videos)} frames (from real_test/0_video/)")
```

### Step 3: Execute Frame Extraction
```bash
cd /workspace/ForgeLens
python scripts/extract_real_test_video_frames.py
```

**Expected console output**:
```
================================================================================
EXTRACTING FRAMES FROM REAL TEST VIDEOS (0_video/)
================================================================================

real_20250326_Youtube: Processing...
  Videos: 150
  Frames extracted: 1,200

real_outputs_20250408_refine: Processing...
  Videos: 40
  Frames extracted: 320

FF++C23: Processing...
  Videos: 21
  Frames extracted: 168

================================================================================
EXTRACTION COMPLETE
================================================================================
Total videos: 211
Total frames: 1,688
```

### Step 4: Regenerate test.json
```bash
cd /workspace/ForgeLens
python scripts/generate_dataset_json.py
```

**Expected changes**:
- **Previous test.json**: 1,483 samples (Real: 369, Fake: 1,114)
- **New test.json**: ~3,171 samples (Real: 2,057, Fake: 1,114)
- **Added**: 1,688 real frames from 211 videos

---

## 5. Expected Impact

### Test Set Statistics (Before → After)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total samples | 1,483 | 3,171 | +1,688 |
| Real samples | 369 | 2,057 | +1,688 |
| Fake samples | 1,114 | 1,114 | 0 |
| Real:Fake ratio | 1:3.0 | 1.85:1 | More balanced |

### Real Test Sources (After)

| Source | Type | Count |
|--------|------|-------|
| real_test_0_real | Images | 246 |
| real_test_06_car_model | Images | 71 |
| real_test_2_뷰티픽 | Images | 52 |
| real_test_0_video_real_20250326_Youtube | Video frames | ~1,200 |
| real_test_0_video_real_outputs_20250408_refine | Video frames | ~320 |
| real_test_0_video_FF++C23 | Video frames | ~168 |
| **Total** | | **~2,057** |

### Evaluation Benefits
1. **More comprehensive real sample coverage** (videos + images)
2. **Better balance between real and fake** (from 1:3 to 1.85:1)
3. **Diverse real sources** for robustness testing
4. **Per-source evaluation** will show performance on video-based real samples

---

## 6. File Structure After Implementation

```
/data/data/deepfake_finetune_dataset/
├── real_test/
│   ├── 0_real/
│   │   └── 1_image/              # Already processed
│   ├── 0_video/
│   │   └── 0_real/
│   │       ├── real_20250326_Youtube/
│   │       │   ├── frames/       # NEW: Extracted frames
│   │       │   │   ├── 009222_frame_000015.png
│   │       │   │   ├── 009222_frame_000030.png
│   │       │   │   └── ...
│   │       │   ├── 009222.mp4
│   │       │   └── ...
│   │       ├── real_outputs_20250408_refine/
│   │       │   ├── frames/       # NEW: Extracted frames
│   │       │   └── ...
│   │       └── FF++C23/
│   │           ├── frames/       # NEW: Extracted frames
│   │           └── ...
│   ├── 06_car_model/             # Already processed
│   └── 2_뷰티픽/                  # Already processed
└── jsons/
    └── test.json                 # UPDATED: Now includes 0_video frames
```

---

## 7. Testing and Validation

After implementation, verify:

1. **Frame extraction success**:
   ```bash
   find /data/data/deepfake_finetune_dataset/real_test/0_video -name "*.png" | wc -l
   # Expected: 1,688
   ```

2. **test.json metadata**:
   ```bash
   python -c "
   import json
   with open('/data/data/deepfake_finetune_dataset/jsons/test.json') as f:
       data = json.load(f)
   print(f\"Total: {data['metadata']['total_samples']}\")
   print(f\"Real: {data['metadata']['real_count']}\")
   print(f\"Fake: {data['metadata']['fake_count']}\")
   "
   # Expected: Total: 3171, Real: 2057, Fake: 1114
   ```

3. **Per-source breakdown**:
   ```bash
   python evaluate_json.py \
     --eval_stage 1 \
     --weights ./check_points/my_experiment/train_stage_1/model/intermediate_model_best.pth \
     --test_json /data/data/deepfake_finetune_dataset/jsons/test.json \
     --experiment_name my_experiment \
     --batch_size 16
   ```
   Should show new sources: `real_test_0_video_real_20250326_Youtube`, etc.

---

## 8. Potential Issues and Solutions

### Issue 1: Video codec compatibility
- **Problem**: Some videos may have codecs not supported by OpenCV
- **Solution**: Add error handling to skip problematic videos and log them

### Issue 2: Disk space
- **Problem**: 1,688 frames × ~200KB = ~337MB additional storage
- **Solution**: Verify available disk space before extraction

### Issue 3: Processing time
- **Problem**: 211 videos might take 10-20 minutes to process
- **Solution**: Add progress bar with `tqdm` for visibility

### Issue 4: JSON file size
- **Problem**: test.json will grow from ~200KB to ~400KB
- **Solution**: No issue, JSON is efficient for this scale

---

## 9. Implementation Checklist

- [ ] Create `extract_real_test_video_frames.py`
- [ ] Test frame extraction on 1-2 sample videos
- [ ] Run full frame extraction on all 211 videos
- [ ] Verify frame count (should be 1,688)
- [ ] Modify `generate_dataset_json.py` to include 0_video frames
- [ ] Regenerate test.json
- [ ] Verify test.json metadata (Real: 2,057, Total: 3,171)
- [ ] Test evaluation script with new test.json
- [ ] Verify per-source evaluation shows new sources
- [ ] Document any issues encountered

---

## 10. Rollback Plan

If issues occur:

1. **Backup current test.json**:
   ```bash
   cp /data/data/deepfake_finetune_dataset/jsons/test.json \
      /data/data/deepfake_finetune_dataset/jsons/test.json.backup
   ```

2. **Keep original generate_dataset_json.py**:
   ```bash
   cp scripts/generate_dataset_json.py scripts/generate_dataset_json.py.backup
   ```

3. **Restore if needed**:
   ```bash
   cp /data/data/deepfake_finetune_dataset/jsons/test.json.backup \
      /data/data/deepfake_finetune_dataset/jsons/test.json
   ```

---

## Summary

This implementation will:
1. Extract 1,688 frames from 211 real videos in `real_test/0_video/`
2. Add them to test.json with proper metadata
3. Increase real test samples from 369 to 2,057
4. Improve real:fake balance from 1:3 to 1.85:1
5. Enable more comprehensive evaluation on video-based real samples

**Estimated implementation time**: 30-45 minutes (including frame extraction)
