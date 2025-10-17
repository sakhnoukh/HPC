# CIFAR-10 Dataset

## Overview
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

## Classes
1. airplane
2. automobile
3. bird
4. cat
5. deer
6. dog
7. frog
8. horse
9. ship
10. truck

## Download
Run the fetch script to download the dataset:
```bash
python data/fetch_cifar10.py --data-dir ./data
```

The dataset will be saved to `./data/cifar-10-batches-py/` (~170 MB).

## Statistics
- **Training samples:** 50,000
- **Test samples:** 10,000
- **Image size:** 32×32 pixels, RGB (3 channels)
- **Format:** Python pickle files
- **Total size:** ~170 MB

## Data Augmentation
The training pipeline uses standard augmentation:
- Random crop (32×32 with padding=4)
- Random horizontal flip
- Normalization (mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

## Citation
Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
https://www.cs.toronto.edu/~kriz/cifar.html
