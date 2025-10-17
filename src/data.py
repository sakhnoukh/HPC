"""
CIFAR-10 data loading and preprocessing utilities.
"""
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


def get_cifar10_transforms(augment=True):
    """
    Get CIFAR-10 transforms for training and validation.
    
    Args:
        augment: Whether to apply data augmentation for training
        
    Returns:
        train_transform, val_transform
    """
    # CIFAR-10 normalization statistics
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
    
    if augment:
        # Training: with augmentation
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # Training: without augmentation (for ablation studies)
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    
    # Validation: no augmentation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    return train_transform, val_transform


def get_cifar10_dataloaders(
    data_dir,
    batch_size,
    num_workers=4,
    distributed=False,
    augment=True,
    pin_memory=True,
    persistent_workers=False
):
    """
    Create CIFAR-10 train and validation dataloaders.
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size per GPU/process
        num_workers: Number of dataloader workers
        distributed: Whether to use DistributedSampler
        augment: Whether to apply data augmentation
        pin_memory: Whether to pin memory for faster GPU transfer
        persistent_workers: Keep workers alive between epochs
        
    Returns:
        train_loader, val_loader
    """
    train_transform, val_transform = get_cifar10_transforms(augment)
    
    # Datasets
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=train_transform
    )
    
    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=val_transform
    )
    
    # Samplers
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            shuffle=True,
            drop_last=False
        )
        val_sampler = DistributedSampler(
            val_dataset,
            shuffle=False,
            drop_last=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # Shuffle only if not distributed
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=False
    )
    
    return train_loader, val_loader


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
