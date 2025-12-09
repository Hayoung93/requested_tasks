"""
Custom Dataset for JSON-based data loading in ForgeLens

This module provides a PyTorch Dataset class that loads image paths and labels
from JSON files, enabling flexible dataset organization without requiring
specific folder structures.

Author: ForgeLens Team
Date: 2025-11-28
"""

import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class JSONDataset(Dataset):
    """
    Custom Dataset for loading images from JSON file

    JSON file format:
    {
        "metadata": {
            "split": "train",
            "total_samples": 1000,
            "real_count": 500,
            "fake_count": 500
        },
        "data": [
            {
                "image_path": "/path/to/image.png",
                "label": 0,  # 0 for real, 1 for fake
                "label_name": "real",
                "source": "dataset_name",
                "video_id": "video_001"
            },
            ...
        ]
    }
    """

    def __init__(self, json_path, transform=None):
        """
        Args:
            json_path (str): Path to JSON file containing dataset information
            transform (callable, optional): Optional transform to be applied on images
        """
        self.json_path = json_path
        self.transform = transform

        # Load JSON file
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, 'r') as f:
            self.data_dict = json.load(f)

        # Extract data and metadata
        self.data = self.data_dict['data']
        self.metadata = self.data_dict.get('metadata', {})

        # Create targets list for WeightedRandomSampler compatibility
        self.targets = [item['label'] for item in self.data]

        print(f"Loaded JSONDataset from {json_path}")
        print(f"  Split: {self.metadata.get('split', 'unknown')}")
        print(f"  Total samples: {len(self.data)}")
        print(f"  Real: {self.metadata.get('real_count', 'unknown')}")
        print(f"  Fake: {self.metadata.get('fake_count', 'unknown')}")

    def __len__(self):
        """Return the total number of samples"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Load and return a sample from the dataset

        Args:
            idx (int): Index of the sample to load

        Returns:
            tuple: (image, label) where image is a transformed PIL Image or Tensor
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")

        item = self.data[idx]

        # Load image
        image_path = item['image_path']

        # Check if image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {str(e)}")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get label
        label = item['label']

        return image, label

    def get_metadata(self):
        """Return metadata from JSON file"""
        return self.metadata

    def get_sample_info(self, idx):
        """
        Get full information about a sample (for debugging/inspection)

        Args:
            idx (int): Index of the sample

        Returns:
            dict: Full item dictionary with all metadata
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        return self.data[idx]

    def get_class_distribution(self):
        """
        Get distribution of classes in the dataset

        Returns:
            dict: Dictionary with class counts
        """
        real_count = sum(1 for item in self.data if item['label'] == 0)
        fake_count = sum(1 for item in self.data if item['label'] == 1)

        return {
            'real': real_count,
            'fake': fake_count,
            'total': len(self.data),
            'ratio': f"{real_count}:{fake_count}"
        }

    def get_source_distribution(self):
        """
        Get distribution of data sources in the dataset

        Returns:
            dict: Dictionary with source counts
        """
        from collections import Counter
        sources = [item['source'] for item in self.data]
        return dict(Counter(sources))

    def __repr__(self):
        """String representation of the dataset"""
        return (f"JSONDataset(json_path='{self.json_path}', "
                f"samples={len(self.data)}, "
                f"split='{self.metadata.get('split', 'unknown')}')")


def test_json_dataset():
    """Test function to verify JSONDataset loading"""
    import torchvision.transforms as transforms

    # Define transforms (same as ForgeLens training)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    # Test with train.json
    json_path = '/data/data/deepfake_finetune_dataset/jsons/train.json'

    if os.path.exists(json_path):
        print("Testing JSONDataset...")
        dataset = JSONDataset(json_path, transform=transform)

        print(f"\nDataset info:")
        print(f"  {dataset}")

        print(f"\nClass distribution:")
        print(f"  {dataset.get_class_distribution()}")

        print(f"\nSource distribution:")
        source_dist = dataset.get_source_distribution()
        for source, count in sorted(source_dist.items()):
            print(f"  {source}: {count}")

        # Load first sample
        print(f"\nLoading first sample...")
        image, label = dataset[0]
        print(f"  Image shape: {image.shape}")
        print(f"  Label: {label} ({'fake' if label == 1 else 'real'})")

        # Get sample info
        info = dataset.get_sample_info(0)
        print(f"  Source: {info['source']}")
        print(f"  Video ID: {info['video_id']}")
        print(f"  Path: {info['image_path']}")

        print("\nTest passed!")
    else:
        print(f"JSON file not found: {json_path}")
        print("Please run generate_dataset_json.py first.")


if __name__ == '__main__':
    test_json_dataset()
