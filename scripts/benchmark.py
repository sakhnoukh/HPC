#!/usr/bin/env python3
"""
Benchmark script to establish baseline performance.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --batch-size 128 --iterations 100
"""
import argparse
import time

import torch
import torch.nn as nn
from torchvision import models

# Try to import from src
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import AverageMeter


def benchmark_forward_backward(model, batch_size, num_iterations, device, dtype=torch.float32):
    """Benchmark forward and backward passes."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # Dummy data
    inputs = torch.randn(batch_size, 3, 32, 32, device=device, dtype=dtype)
    targets = torch.randint(0, 10, (batch_size,), device=device)
    
    # Warmup
    for _ in range(10):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Benchmark
    forward_time = AverageMeter('Forward')
    backward_time = AverageMeter('Backward')
    total_time = AverageMeter('Total')
    
    for _ in range(num_iterations):
        # Forward
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        forward_time.update(time.time() - start)
        
        # Backward
        start = time.time()
        loss.backward()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        backward_time.update(time.time() - start)
        
        total_time.update(forward_time.val + backward_time.val)
    
    return forward_time.avg, backward_time.avg, total_time.avg


def benchmark_dataloader(data_dir, batch_size, num_workers):
    """Benchmark dataloader performance."""
    from src.data import get_cifar10_dataloaders
    
    train_loader, _ = get_cifar10_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=False,
        augment=True,
        pin_memory=True
    )
    
    # Warmup
    for i, (images, targets) in enumerate(train_loader):
        if i >= 5:
            break
    
    # Benchmark
    start = time.time()
    total_batches = 0
    for images, targets in train_loader:
        total_batches += 1
        if total_batches >= 100:
            break
    elapsed = time.time() - start
    
    batches_per_sec = total_batches / elapsed
    images_per_sec = batches_per_sec * batch_size
    
    return batches_per_sec, images_per_sec


def main():
    parser = argparse.ArgumentParser(description='Benchmark performance')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for benchmarking')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of iterations')
    parser.add_argument('--data', type=str, default='./data',
                        help='Path to CIFAR-10 data')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')
    args = parser.parse_args()
    
    print("=" * 70)
    print("CIFAR-10 ResNet-18 Benchmark")
    print("=" * 70)
    print()
    
    # Device info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Iterations: {args.iterations}")
    print()
    
    # Create model
    print("Creating ResNet-18...")
    model = models.resnet18(num_classes=10).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    print()
    
    # Benchmark FP32
    print("-" * 70)
    print("Benchmarking FP32")
    print("-" * 70)
    fwd, bwd, total = benchmark_forward_backward(
        model, args.batch_size, args.iterations, device, dtype=torch.float32
    )
    throughput_fp32 = args.batch_size / total
    
    print(f"Forward:  {fwd*1000:.2f} ms")
    print(f"Backward: {bwd*1000:.2f} ms")
    print(f"Total:    {total*1000:.2f} ms")
    print(f"Throughput: {throughput_fp32:.0f} images/sec")
    print()
    
    # Benchmark FP16/BF16 if CUDA available
    if torch.cuda.is_available():
        # BF16
        if torch.cuda.is_bf16_supported():
            print("-" * 70)
            print("Benchmarking BF16")
            print("-" * 70)
            
            model_bf16 = models.resnet18(num_classes=10).to(device).to(torch.bfloat16)
            fwd, bwd, total = benchmark_forward_backward(
                model_bf16, args.batch_size, args.iterations, device, dtype=torch.bfloat16
            )
            throughput_bf16 = args.batch_size / total
            speedup = throughput_bf16 / throughput_fp32
            
            print(f"Forward:  {fwd*1000:.2f} ms")
            print(f"Backward: {bwd*1000:.2f} ms")
            print(f"Total:    {total*1000:.2f} ms")
            print(f"Throughput: {throughput_bf16:.0f} images/sec ({speedup:.2f}x vs FP32)")
            print()
        else:
            print("BF16 not supported on this GPU")
            print()
        
        # FP16
        print("-" * 70)
        print("Benchmarking FP16 (with AMP)")
        print("-" * 70)
        
        model_fp16 = models.resnet18(num_classes=10).to(device)
        fwd, bwd, total = benchmark_forward_backward(
            model_fp16, args.batch_size, args.iterations, device, dtype=torch.float16
        )
        throughput_fp16 = args.batch_size / total
        speedup = throughput_fp16 / throughput_fp32
        
        print(f"Forward:  {fwd*1000:.2f} ms")
        print(f"Backward: {bwd*1000:.2f} ms")
        print(f"Total:    {total*1000:.2f} ms")
        print(f"Throughput: {throughput_fp16:.0f} images/sec ({speedup:.2f}x vs FP32)")
        print()
    
    # Benchmark DataLoader if data available
    if os.path.exists(args.data):
        print("-" * 70)
        print("Benchmarking DataLoader")
        print("-" * 70)
        
        try:
            batches_per_sec, images_per_sec = benchmark_dataloader(
                args.data, args.batch_size, args.num_workers
            )
            print(f"Workers: {args.num_workers}")
            print(f"Batches/sec: {batches_per_sec:.1f}")
            print(f"Images/sec: {images_per_sec:.0f}")
            print()
        except Exception as e:
            print(f"DataLoader benchmark failed: {e}")
            print("Make sure CIFAR-10 is downloaded: python data/fetch_cifar10.py")
            print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"FP32 Throughput: {throughput_fp32:.0f} images/sec")
    
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            print(f"BF16 Throughput: {throughput_bf16:.0f} images/sec ({throughput_bf16/throughput_fp32:.2f}x)")
        print(f"FP16 Throughput: {throughput_fp16:.0f} images/sec ({throughput_fp16/throughput_fp32:.2f}x)")
    
    print()
    print("Recommendation:")
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print("  Use --precision bf16 for best performance on this GPU")
    elif torch.cuda.is_available():
        print("  Use --precision fp16 for better performance")
    else:
        print("  Running on CPU - consider using a GPU for better performance")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
