# YouTube Real Data Addition Plan

**Date**: 2025-12-04
**Objective**: Add youtube_real data to training set to improve real data diversity and reduce OOD generalization issues

## Problem Analysis

### Current Issue
- Training data contains ONLY klleon_real (YouTube celebrity videos)
- Test set contains completely different real data sources:
  - car_real: 80% error rate (OOD)
  - 폴리곰: 50% error rate (OOD)
  - 뷰티픽: 4% error rate
- Overall false positive rate: 43.63% (161/369 real samples)
- 100% of false positives are from OOD data

### Root Cause
```
Training Real Data:
  ✅ klleon_real (YouTube celebrities) - 8,437 frames (100%)
  ❌ youtube_real (106 videos)          - 0 frames (UNUSED!)

Result: Model overfits to klleon_real style
→ High false positive on unseen real data distributions
```

## Solution: Add youtube_real to Training Set

### Frame Extraction Parameters (Match klleon_real)
- **Frames per video**: 13
- **Sampling method**: Even intervals using numpy.linspace
- **Naming convention**: `{video_name}_frame_{idx:06d}.png`
- **Output location**: `/data/data/deepfake_finetune_dataset/real/youtube_real/frames/`
- **Total frames**: 106 videos × 13 frames = ~1,378 frames

### JSON Modification Strategy

#### train_small_5000.json
**Before**:
- Total: 5,000 samples
- Real: 2,500 (klleon_real 100%)
- Fake: 2,500

**After**:
- Total: 5,000 samples
- Real: 2,500
  - klleon_real: 1,250 (50%)
  - youtube_real: 1,250 (50%)
- Fake: 2,500

#### train.json
**Before**:
- Total: 17,349 samples
- Real: 8,437 (klleon_real 100%)
- Fake: 8,912

**After**:
- Total: ~14,508 samples
- Real: ~5,596
  - klleon_real: 4,218 (50% of original)
  - youtube_real: 1,378 (all available)
- Fake: 8,912

**Note**: train.json total will decrease but diversity increases significantly

## Implementation Steps

### Step 1: Document Creation ✓
- Create this plan document in plans/

### Step 2: Frame Extraction
**Script**: `scripts/extract_youtube_real_frames.py`

```python
# Extract 13 frames per video from youtube_real
# Use even interval sampling (numpy.linspace)
# Save to youtube_real/frames/
```

**Expected Output**:
- ~1,378 PNG files in `/data/data/deepfake_finetune_dataset/real/youtube_real/frames/`

### Step 3: JSON Regeneration
**Script**: `scripts/regenerate_train_jsons_with_youtube.py`

**Logic**:
1. Load existing train.json and train_small_5000.json
2. Extract klleon_real samples
3. Randomly select 50% of klleon_real (seed=3407)
4. Add youtube_real frames
5. Shuffle and save new JSONs
6. Backup original JSONs

**Validation**:
- Check real/fake ratio
- Verify source distribution
- Validate all file paths exist

### Step 4: Verification
- Count extracted frames
- Verify JSON integrity
- Check source balance
- Test loading with JSONDataset

## Expected Results

### Improved Diversity
- Real data sources: 1 → 2 (100% increase)
- Real data distribution more representative

### Expected Improvements
- Lower false positive rate on OOD data
- Better generalization to unseen real data
- Maintained fake detection performance

### Metrics to Track
- Overall accuracy
- False positive rate (real → fake)
- Per-source accuracy (car_real, 폴리곰, etc.)

## Backup Strategy
- Original JSONs backed up to `jsons_backup/`
- Can rollback if needed

## Notes
- Use seed=3407 for reproducibility
- Maintain video-level split (all frames from same video in same split)
- Keep fake data unchanged

## Execution Log

### 2025-12-04 - Initial Planning
- Analyzed OOD problem
- Confirmed youtube_real directory exists but unused
- Designed 50% replacement strategy
- Created this plan document
