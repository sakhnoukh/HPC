#!/usr/bin/env python3
"""
Download CIFAR-10 dataset for training.

Usage:
    python data/fetch_cifar10.py --data-dir ./data
"""
import argparse
import os
from torchvision import datasets


def main():
    parser = argparse.ArgumentParser(description='Download CIFAR-10 dataset')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory to download dataset (default: ./data)')
    args = parser.parse_args()
    
    print(f"Downloading CIFAR-10 to {args.data_dir}...")
    print("=" * 60)
    
    # Training set
    print("\n[1/2] Downloading training set...")
    train_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True
    )
    print(f"✓ Training samples: {len(train_dataset)}")
    
    # Test set
    print("\n[2/2] Downloading test set...")
    test_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True
    )
    print(f"✓ Test samples: {len(test_dataset)}")
    
    print("\n" + "=" * 60)
    print("✓ Download complete!")
    print(f"Dataset location: {os.path.join(args.data_dir, 'cifar-10-batches-py')}")
    print(f"Total size: ~170 MB")
    print("\nDataset info:")
    print("  - 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck")
    print("  - Image size: 32x32 RGB")
    print("  - Training: 50,000 images")
    print("  - Test: 10,000 images")


if __name__ == '__main__':
    main()
