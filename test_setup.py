#!/usr/bin/env python3
"""
Quick test script to verify environment setup.

Usage:
    python test_setup.py
"""
import sys

def test_imports():
    """Test that all required packages are available."""
    print("Testing package imports...")
    print("-" * 60)
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'torchvision',
        'numpy': 'NumPy',
        'pandas': 'Pandas (optional)',
        'matplotlib': 'Matplotlib (optional)',
    }
    
    success = True
    for module, name in required.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}: NOT FOUND")
            if module in ['torch', 'torchvision', 'numpy']:
                success = False
    
    return success


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    print("-" * 60)
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {props.name}")
                print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
                print(f"  Compute Capability: {props.major}.{props.minor}")
        else:
            print("Running on CPU (no CUDA available)")
        
        return True
    except Exception as e:
        print(f"✗ Error testing CUDA: {e}")
        return False


def test_distributed():
    """Test distributed training support."""
    print("\nTesting Distributed Training Support...")
    print("-" * 60)
    
    try:
        import torch
        
        nccl_available = torch.distributed.is_nccl_available()
        gloo_available = torch.distributed.is_gloo_available()
        
        print(f"NCCL backend available: {nccl_available}")
        print(f"Gloo backend available: {gloo_available}")
        
        if nccl_available:
            print("✓ Ready for multi-GPU training")
        elif gloo_available:
            print("! Only Gloo backend available (CPU-only distributed)")
        else:
            print("✗ No distributed backends available")
        
        return True
    except Exception as e:
        print(f"✗ Error testing distributed: {e}")
        return False


def test_data_loading():
    """Test that data module works."""
    print("\nTesting Data Loading...")
    print("-" * 60)
    
    try:
        from src.data import get_cifar10_transforms
        
        train_transform, val_transform = get_cifar10_transforms()
        print("✓ Data transforms created successfully")
        
        return True
    except Exception as e:
        print(f"✗ Error testing data loading: {e}")
        return False


def test_utilities():
    """Test utility functions."""
    print("\nTesting Utilities...")
    print("-" * 60)
    
    try:
        from src.utils import set_seed, Timer, AverageMeter, get_git_hash
        
        # Test seeding
        set_seed(42)
        print("✓ Seeding works")
        
        # Test timer
        timer = Timer()
        timer.start()
        import time
        time.sleep(0.01)
        elapsed = timer.stop()
        print(f"✓ Timer works (measured {elapsed:.3f}s)")
        
        # Test average meter
        meter = AverageMeter('test')
        meter.update(1.0)
        meter.update(2.0)
        print(f"✓ AverageMeter works (avg={meter.avg})")
        
        # Test git hash
        git_hash = get_git_hash()
        print(f"✓ Git hash: {git_hash}")
        
        return True
    except Exception as e:
        print(f"✗ Error testing utilities: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("HPC CIFAR-10 Setup Verification")
    print("=" * 60)
    
    results = []
    
    results.append(("Package Imports", test_imports()))
    results.append(("CUDA Support", test_cuda()))
    results.append(("Distributed Training", test_distributed()))
    results.append(("Data Loading", test_data_loading()))
    results.append(("Utilities", test_utilities()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed! Environment is ready.")
        print("\nNext steps:")
        print("  1. Download CIFAR-10: python data/fetch_cifar10.py")
        print("  2. Test training: python src/train.py --epochs 1 --batch-size 64")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix issues before proceeding.")
        print("\nTroubleshooting:")
        print("  - Install missing packages: pip install -r env/requirements.txt")
        print("  - Check CUDA installation if GPU tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
