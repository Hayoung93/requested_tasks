# ForgeLens Custom Dataset Fine-tuning Implementation Plan

## 📋 Overview

This document outlines the implementation plan for fine-tuning ForgeLens on a custom deepfake dataset with irregular folder structures. The implementation uses JSON-based data loading to handle diverse data organization patterns.

## 🎯 Project Goals

1. Process custom deepfake dataset from `/data/data/deepfake_finetune_dataset/`
2. Extract frames from video files with even interval sampling
3. Create balanced train/val/test splits
4. Implement JSON-based data loading system
5. Maintain compatibility with existing ForgeLens training pipeline

## 📊 Dataset Specifications

### Source Data Location
- **Raw Dataset**: `/data/data/deepfake_finetune_dataset/`
- **JSON Output**: `/data/data/deepfake_finetune_dataset/jsons/`

### Dataset Sources

#### Fake/AI-Generated Images
1. **kling2.5_dataset/** - 3,840 frames (already extracted)
   - Structure: `kling2.5_dataset/{video_id}/frames/{frame}.png`
   - 62 video directories

2. **veo3_dataset/** - 6,400 frames (already extracted)
   - Structure: `veo3_dataset/{video_id}/frames/{frame}.png`
   - 102 video directories

3. **nanobanana_dataset/** - 100 images
   - Structure: `nanobanana_dataset/{frame_id}.png`
   - Flat directory

4. **sora_dataset/** - 100 MP4 videos
   - Structure: `sora_dataset/{video_id}.mp4`
   - Requires frame extraction: 8 frames/video (even interval)
   - Expected output: 800 frames

**Total Fake Images**: ~11,140 frames

#### Real Images

1. **real/** folder - MP4 videos
   - Structure: `real/1. klleon_real_dataset/{person_name}/{video}.mp4`
   - Requires frame extraction with calculated frame count to balance with Fake class

2. **real_test/** folder - Organized test images
   - Structure: `real_test/{category}/{class_label}/{source}/images`
   - Used for test set only

**Target Real Images**: ~11,140 frames (1:1 balance with Fake)

### Data Split Strategy

#### Fake Class
- **Split Ratio**: Train/Val/Test = 8:1:1
- **Split Unit**: Video-level (all frames from same video stay together)
- **Per-Source Split**: Each dataset source (kling2.5, veo3, nanobanana, sora) is split independently

#### Real Class
- **Train/Val**: Split real/ folder videos 8:2
- **Test**: Use all images from real_test/ folder
- **Split Unit**: Video-level for real/ folder

#### Class Balance
- Maintain 1:1 Real:Fake ratio in train/val sets
- Use WeightedRandomSampler during training for perfect balance

## 🏗️ Implementation Architecture

### Directory Structure
```
/workspace/ForgeLens/
├── plans/
│   └── custom_dataset_implementation.md    # This document
├── scripts/
│   ├── analyze_dataset.py                  # Dataset analysis and statistics
│   ├── extract_frames.py                   # Video frame extraction
│   └── generate_dataset_json.py            # JSON file generation
├── dataset.py                              # Custom JSONDataset class
├── util.py                                 # Updated with get_dataset_from_json()
├── options.py                              # Updated with JSON path arguments
└── train.py                                # Updated for JSON-based loading

/data/data/deepfake_finetune_dataset/
├── kling2.5_dataset/
├── veo3_dataset/
├── nanobanana_dataset/
├── sora_dataset/
│   ├── {video_id}.mp4
│   └── frames/                             # Created by extract_frames.py
│       └── {video_id}_frame_{idx}.png
├── real/
│   └── 1. klleon_real_dataset/
│       └── {person_name}/
│           ├── {video}.mp4
│           └── frames/                     # Created by extract_frames.py
│               └── {video}_frame_{idx}.png
├── real_test/
└── jsons/
    ├── train.json                          # Training split
    ├── val.json                            # Validation split
    └── test.json                           # Test split
```

### JSON Schema

```json
{
  "metadata": {
    "split": "train",
    "created_date": "2025-11-28",
    "total_samples": 10000,
    "real_count": 5000,
    "fake_count": 5000
  },
  "data": [
    {
      "image_path": "/data/data/deepfake_finetune_dataset/sora_dataset/frames/video_001_frame_0042.png",
      "label": 1,
      "label_name": "fake",
      "source": "sora",
      "video_id": "video_001",
      "frame_idx": 42
    },
    {
      "image_path": "/data/data/deepfake_finetune_dataset/real/1. klleon_real_dataset/aaronpaul/frames/video_01_frame_0010.png",
      "label": 0,
      "label_name": "real",
      "source": "klleon_real",
      "video_id": "video_01",
      "frame_idx": 10
    }
  ]
}
```

## 🔧 Implementation Steps

### Step 1: Create Plans and Scripts Folders ✓
```bash
mkdir -p /workspace/ForgeLens/plans
mkdir -p /workspace/ForgeLens/scripts
```

### Step 2: Analyze Dataset
**Script**: `scripts/analyze_dataset.py`

**Purpose**:
- Count total videos and existing frames
- Calculate required frames per real video for 1:1 balance
- Generate statistics report

**Output**:
```
Dataset Analysis Report
=======================
Fake Class:
  - kling2.5: 3,840 frames (62 videos)
  - veo3: 6,400 frames (102 videos)
  - nanobanana: 100 frames
  - sora: 100 videos (800 frames expected)
  Total Fake: 11,140 frames

Real Class:
  - real/ videos: X videos
  - Frames per video needed: Y
  - Total Real (train/val): ~11,140 frames
  - real_test/ images: Z frames
```

### Step 3: Extract Frames
**Script**: `scripts/extract_frames.py`

**Functionality**:
1. Extract frames from sora_dataset/ videos
   - 8 frames per video with even interval
   - Save as: `sora_dataset/frames/{video_id}_frame_{original_idx}.png`

2. Extract frames from real/ videos
   - Calculated number per video for 1:1 balance
   - Save as: `real/{path}/frames/{video_name}_frame_{original_idx}.png`

3. Handle edge cases:
   - Videos shorter than required frames: extract available frames and report
   - Skip corrupted videos with error logging

**Frame Extraction Logic**:
```python
def extract_frames_even_interval(video_path, output_dir, num_frames=8):
    """Extract frames at even intervals"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        print(f"WARNING: {video_path} has only {total_frames} frames")
        num_frames = total_frames

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_path = f"{output_dir}/{video_name}_frame_{idx:06d}.png"
            cv2.imwrite(frame_path, frame)
```

### Step 4: Generate JSON Files
**Script**: `scripts/generate_dataset_json.py`

**Functionality**:
1. Scan all extracted frames and existing images
2. Group frames by video ID
3. Split videos (not individual frames) into train/val/test
4. Generate train.json, val.json, test.json

**Split Logic**:
```python
# Fake class: 8:1:1 per source
for source in ['kling2.5', 'veo3', 'nanobanana', 'sora']:
    videos = get_videos(source)
    random.shuffle(videos)

    train_videos = videos[:int(0.8 * len(videos))]
    val_videos = videos[int(0.8 * len(videos)):int(0.9 * len(videos))]
    test_videos = videos[int(0.9 * len(videos)):]

# Real class: 8:2 for train/val, real_test for test
real_videos = get_videos('real/')
random.shuffle(real_videos)

train_videos = real_videos[:int(0.8 * len(real_videos))]
val_videos = real_videos[int(0.8 * len(real_videos)):]
test_images = get_images('real_test/')
```

### Step 5: Implement JSONDataset Class
**File**: `dataset.py`

```python
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class JSONDataset(Dataset):
    """Custom Dataset for loading images from JSON file"""

    def __init__(self, json_path, transform=None):
        """
        Args:
            json_path (str): Path to JSON file
            transform (callable): Optional transform to be applied on images
        """
        with open(json_path, 'r') as f:
            self.data_dict = json.load(f)

        self.data = self.data_dict['data']
        self.transform = transform
        self.targets = [item['label'] for item in self.data]  # For WeightedRandomSampler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        image_path = item['image_path']
        image = Image.open(image_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = item['label']

        return image, label

    def get_metadata(self):
        """Return metadata from JSON"""
        return self.data_dict['metadata']
```

### Step 6: Update util.py
**File**: `util.py`

Add new function:
```python
def get_dataset_from_json(json_path, transform=None):
    """
    Load dataset from JSON file

    Args:
        json_path (str): Path to JSON file
        transform (callable): Transform to apply to images

    Returns:
        JSONDataset: Dataset object
    """
    from dataset import JSONDataset

    if transform is None:
        transform = transforms.Compose([
            translate_duplicate,
            transforms.RandomCrop([224, 224]),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

    dataset = JSONDataset(json_path, transform=transform)
    return dataset

def get_dataset_from_json_test(json_path):
    """
    Load test dataset from JSON file (with CenterCrop instead of RandomCrop)
    """
    transform = transforms.Compose([
        translate_duplicate,
        transforms.CenterCrop([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    from dataset import JSONDataset
    dataset = JSONDataset(json_path, transform=transform)
    return dataset
```

### Step 7: Update options.py
**File**: `options/options.py`

Add new arguments:
```python
# JSON dataset paths
parser.add_argument('--use_json_dataset', action='store_true',
                    help='Use JSON-based dataset loading instead of ImageFolder')
parser.add_argument('--train_json', type=str, default='/data/data/deepfake_finetune_dataset/jsons/train.json',
                    help='Path to training JSON file')
parser.add_argument('--val_json', type=str, default='/data/data/deepfake_finetune_dataset/jsons/val.json',
                    help='Path to validation JSON file')
parser.add_argument('--test_json', type=str, default='/data/data/deepfake_finetune_dataset/jsons/test.json',
                    help='Path to test JSON file')
```

### Step 8: Update train.py
**File**: `train.py`

Modify data loading section:
```python
# Load dataset
if opt.use_json_dataset:
    print(f"Loading JSON-based dataset...")
    print(f"  Train: {opt.train_json}")
    print(f"  Val: {opt.val_json}")

    train_dataset = get_dataset_from_json(opt.train_json)
    val_dataset = get_dataset_from_json_test(opt.val_json)
else:
    print(f"Loading ImageFolder-based dataset...")
    train_dataset = get_dataset(opt.train_data_root, opt.train_classes)
    val_dataset = get_dataset_test(opt.val_data_root, opt.val_classes)

# Create balanced sampler
train_sampler = get_bal_sampler(train_dataset)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=opt.stage1_batch_size if opt.training_stage == 1 else opt.stage2_batch_size,
    sampler=train_sampler,
    num_workers=opt.num_workers
)

val_loader = DataLoader(
    val_dataset,
    batch_size=opt.stage1_batch_size if opt.training_stage == 1 else opt.stage2_batch_size,
    shuffle=False,
    num_workers=opt.num_workers
)
```

### Step 9: Testing
**Test Script**: Create `scripts/test_data_loading.py`

```python
from options.options import Options
from util import get_dataset_from_json, get_bal_sampler
from torch.utils.data import DataLoader

# Test data loading
opt = Options().parse()
opt.use_json_dataset = True
opt.train_json = '/data/data/deepfake_finetune_dataset/jsons/train.json'

dataset = get_dataset_from_json(opt.train_json)
print(f"Dataset size: {len(dataset)}")
print(f"Metadata: {dataset.get_metadata()}")

# Test balanced sampler
sampler = get_bal_sampler(dataset)
loader = DataLoader(dataset, batch_size=16, sampler=sampler)

# Load one batch
for images, labels in loader:
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Label distribution: Real={sum(labels==0)}, Fake={sum(labels==1)}")
    break
```

## 📈 Expected Results

### Dataset Statistics
- **Train Set**: ~8,912 images (4,456 real + 4,456 fake)
- **Val Set**: ~1,114 images (557 real + 557 fake)
- **Test Set**: ~1,114 fake + all real_test images

### Training Command
```bash
python train.py \
    --use_json_dataset \
    --train_json /data/data/deepfake_finetune_dataset/jsons/train.json \
    --val_json /data/data/deepfake_finetune_dataset/jsons/val.json \
    --training_stage 1 \
    --stage1_epochs 50 \
    --stage1_batch_size 32 \
    --experiment_name custom_deepfake_finetuning
```

## ⚠️ Important Notes

1. **Video-level splitting**: Ensures frames from the same video never appear in different splits (prevents data leakage)
2. **Even interval sampling**: Maximizes temporal diversity of extracted frames
3. **Class balance**: WeightedRandomSampler ensures 1:1 Real:Fake in each batch
4. **Backward compatibility**: ImageFolder-based loading still works (use `--use_json_dataset` to switch)
5. **Frame naming**: Uses original frame indices for traceability

## 🐛 Troubleshooting

### Short videos
- Script will extract available frames and report warnings
- Check logs for videos with fewer frames than requested

### Missing frames
- Verify video extraction completed successfully
- Check frames/ subdirectories exist

### Imbalanced batches
- Verify `targets` attribute in JSONDataset
- Check WeightedRandomSampler is used

## 📝 File Checklist

- [x] `/workspace/ForgeLens/plans/custom_dataset_implementation.md`
- [ ] `/workspace/ForgeLens/scripts/analyze_dataset.py`
- [ ] `/workspace/ForgeLens/scripts/extract_frames.py`
- [ ] `/workspace/ForgeLens/scripts/generate_dataset_json.py`
- [ ] `/workspace/ForgeLens/dataset.py`
- [ ] `/workspace/ForgeLens/util.py` (modified)
- [ ] `/workspace/ForgeLens/options/options.py` (modified)
- [ ] `/workspace/ForgeLens/train.py` (modified)
- [ ] `/data/data/deepfake_finetune_dataset/jsons/train.json`
- [ ] `/data/data/deepfake_finetune_dataset/jsons/val.json`
- [ ] `/data/data/deepfake_finetune_dataset/jsons/test.json`

## 🎉 Success Criteria

1. All video frames extracted successfully
2. JSON files generated with correct splits
3. Data loader successfully loads batches
4. Class balance maintained in training batches
5. Training runs for at least 1 epoch without errors
