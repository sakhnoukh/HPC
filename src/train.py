#!/usr/bin/env python3
"""
ResNet-18 training on CIFAR-10 with single-GPU or DDP support.

Usage:
    # Single GPU
    python src/train.py --epochs 5 --batch-size 128 --data ./data
    
    # DDP (multi-GPU, single node)
    torchrun --nproc_per_node=4 src/train.py --epochs 5 --batch-size 128 --data ./data
"""
import argparse
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from data import get_cifar10_dataloaders
from utils import (
    set_seed, get_git_hash, get_slurm_job_id, Timer, AverageMeter,
    accuracy, CSVLogger, format_time
)


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='CIFAR-10 ResNet-18 Training')
    
    # Data
    parser.add_argument('--data', type=str, default='./data',
                        help='Path to CIFAR-10 dataset directory')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Use data augmentation')
    
    # Model
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34'],
                        help='Model architecture')
    
    # Training
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'multistep'],
                        help='Learning rate scheduler')
    
    # Mixed Precision
    parser.add_argument('--precision', type=str, default='fp32',
                        choices=['fp32', 'fp16', 'bf16'],
                        help='Training precision (fp32, fp16, or bf16)')
    
    # System
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--print-freq', type=int, default=50,
                        help='Print frequency (batches)')
    
    # Output
    parser.add_argument('--results-dir', type=str, default='./results/csv',
                        help='Directory to save CSV results')
    parser.add_argument('--exp-name', type=str, default='baseline',
                        help='Experiment name for output files')
    
    # DDP (will be set automatically by torchrun)
    parser.add_argument('--local-rank', type=int, default=0,
                        help='Local rank for distributed training')
    
    return parser.parse_args()


def setup_distributed():
    """
    Setup distributed training environment.
    
    Returns:
        rank, world_size, local_rank, is_distributed
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Running with torchrun/torch.distributed.launch
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank, True
    else:
        # Single GPU or CPU
        return 0, 1, 0, False


def train_epoch(train_loader, model, criterion, optimizer, epoch, args, device, rank, scaler=None):
    """Train for one epoch."""
    batch_time = AverageMeter('Time')
    data_time = AverageMeter('Data')
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    
    model.train()
    use_amp = scaler is not None
    
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # Forward pass with mixed precision if enabled
        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16 if args.precision == 'fp16' else torch.bfloat16):
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print progress
        if rank == 0 and i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Acc@1 {top1.val:.2f} ({top1.avg:.2f})')
    
    return losses.avg, top1.avg


def validate(val_loader, model, criterion, device, rank):
    """Evaluate on validation set."""
    batch_time = AverageMeter('Time')
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # Forward pass
            output = model(images)
            loss = criterion(output, target)
            
            # Measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
    if rank == 0:
        print(f' * Acc@1 {top1.avg:.2f}% | Loss {losses.avg:.4f}')
    
    return losses.avg, top1.avg


def main():
    args = get_args()
    
    # Setup distributed training
    rank, world_size, local_rank, is_distributed = setup_distributed()
    
    # Only print from rank 0
    if rank == 0:
        print("=" * 80)
        print("CIFAR-10 ResNet-18 Training")
        print("=" * 80)
        print(f"Distributed: {is_distributed}")
        print(f"World size: {world_size}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Global batch size: {args.batch_size * world_size}")
        print(f"Epochs: {args.epochs}")
        print("=" * 80)
    
    # Set random seed
    set_seed(args.seed + rank)  # Different seed per rank
    
    # Device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
        if rank == 0:
            print("WARNING: CUDA not available, using CPU")
    
    # Data loaders
    train_loader, val_loader = get_cifar10_dataloaders(
        data_dir=args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=is_distributed,
        augment=args.augment,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )
    
    if rank == 0:
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Train batches per epoch: {len(train_loader)}")
    
    # Model
    if args.arch == 'resnet18':
        model = models.resnet18(num_classes=10)
    elif args.arch == 'resnet34':
        model = models.resnet34(num_classes=10)
    
    model = model.to(device)
    
    # Wrap with DDP if distributed
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank
        )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.scheduler == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    
    # Mixed precision scaler
    scaler = None
    if args.precision in ['fp16', 'bf16'] and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler(enabled=(args.precision == 'fp16'))
        if rank == 0:
            print(f"Using mixed precision training: {args.precision}")
    
    # CSV logger (only rank 0)
    if rank == 0:
        csv_fields = [
            'timestamp', 'jobid', 'commit', 'epoch', 'epochs',
            'world_size', 'gpus', 'batch_per_gpu', 'global_batch',
            'train_loss', 'train_acc', 'val_loss', 'val_acc',
            'epoch_time_s', 'images_per_sec'
        ]
        csv_file = os.path.join(
            args.results_dir,
            f'{args.exp_name}_{world_size}gpu_{get_slurm_job_id()}.csv'
        )
        logger = CSVLogger(csv_file, csv_fields)
        
        git_hash = get_git_hash()
        job_id = get_slurm_job_id()
        print(f"Git commit: {git_hash}")
        print(f"Job ID: {job_id}")
        print(f"Logging to: {csv_file}")
        print("=" * 80)
    
    # Training loop
    best_acc = 0.0
    total_start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        # Set epoch for distributed sampler (ensures different shuffle each epoch)
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_acc = train_epoch(
            train_loader, model, criterion, optimizer, epoch, args, device, rank, scaler
        )
        
        # Validate
        val_loss, val_acc = validate(val_loader, model, criterion, device, rank)
        
        # Step scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        images_per_sec = len(train_loader.dataset) / epoch_time * world_size
        
        # Update best accuracy
        if val_acc > best_acc:
            best_acc = val_acc
        
        # Log results (only rank 0)
        if rank == 0:
            print(f"\nEpoch {epoch}/{args.epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Epoch Time: {format_time(epoch_time)} ({epoch_time:.2f}s)")
            print(f"  Throughput: {images_per_sec:.0f} images/sec")
            print(f"  Best Val Acc: {best_acc:.2f}%")
            print("-" * 80)
            
            # Log to CSV
            logger.log({
                'timestamp': datetime.now().isoformat(),
                'jobid': job_id,
                'commit': git_hash,
                'epoch': epoch,
                'epochs': args.epochs,
                'world_size': world_size,
                'gpus': world_size,
                'batch_per_gpu': args.batch_size,
                'global_batch': args.batch_size * world_size,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'epoch_time_s': epoch_time,
                'images_per_sec': images_per_sec
            })
    
    total_time = time.time() - total_start_time
    
    if rank == 0:
        print("=" * 80)
        print(f"Training Complete!")
        print(f"Total Time: {format_time(total_time)}")
        print(f"Best Validation Accuracy: {best_acc:.2f}%")
        print(f"Results saved to: {csv_file}")
        print("=" * 80)
    
    # Cleanup distributed
    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
