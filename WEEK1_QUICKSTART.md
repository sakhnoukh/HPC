# Week 1-2 Quick Start Guide

## 🎯 This Week's Goal
Get a working single-GPU ResNet-18 trainer on CIFAR-10 with CSV logging.

---

## ⚡ Step-by-Step Actions

### 1. Repository Setup (15 minutes)
```bash
cd "/Users/samiakhnoukh/Documents/UNI/Year 3/Semester 1/HPC/HPC App"

# Create directory structure
mkdir -p src/plots env slurm data results/{csv,plots,logs} docs

# Initialize git if not done
git init
git remote add origin https://github.com/sakhnoukh/HPC.git

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/

# Data
data/cifar-10-batches-py/
data/cifar-10-python.tar.gz
*.pth
*.ckpt

# Results (track structure, not outputs)
results/csv/*.csv
results/plots/*.png
results/plots/*.svg
results/logs/*.out
results/logs/*.nsys-rep
results/logs/*.ncu-rep

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
EOF
```

---

### 2. Environment Decision (30 minutes)

You need to choose between **Apptainer** (containerized) or **Modules** (traditional HPC).

#### Option A: Apptainer (Recommended)
**Pros:** Reproducible, portable, isolated  
**Cons:** Requires Apptainer/Singularity on cluster

Create `env/project.def`:
```bash
cat > env/project.def << 'EOF'
Bootstrap: docker
From: nvcr.io/nvidia/pytorch:23.12-py3

%post
    # Install additional dependencies
    pip install --no-cache-dir \
        matplotlib \
        seaborn \
        pandas \
        torchvision

%environment
    export PYTHONUNBUFFERED=1
    export NCCL_DEBUG=INFO

%runscript
    exec python "$@"
EOF
```

**Build container:**
```bash
# On cluster (if you have Apptainer)
apptainer build hpc_pytorch.sif env/project.def
```

#### Option B: Modules (Traditional)
**Pros:** No container build needed  
**Cons:** Version management complexity

Create `env/load_modules.sh`:
```bash
cat > env/load_modules.sh << 'EOF'
#!/bin/bash
# Load required modules for HPC cluster
module purge
module load gcc/11.2.0
module load cuda/12.1
module load openmpi/4.1.5
module load python/3.10
module load pytorch/2.0.1

# Create virtual environment if needed
# python -m venv venv
# source venv/bin/activate
# pip install torchvision matplotlib seaborn pandas

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NCCL_DEBUG=INFO
EOF

chmod +x env/load_modules.sh
```

**Note:** Adjust module names/versions to match your cluster. Run `module avail` on your cluster to see available modules.

---

### 3. Data Fetching Script (20 minutes)

Create `data/fetch_cifar10.py`:
```python
"""
Download CIFAR-10 dataset for training.
Usage: python data/fetch_cifar10.py --data-dir ./data
"""
import argparse
import os
from torchvision import datasets

def main():
    parser = argparse.ArgumentParser(description='Download CIFAR-10 dataset')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory to download dataset')
    args = parser.parse_args()
    
    print(f"Downloading CIFAR-10 to {args.data_dir}...")
    
    # Training set
    datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True
    )
    
    # Test set
    datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True
    )
    
    print("Download complete!")
    print(f"Dataset location: {os.path.join(args.data_dir, 'cifar-10-batches-py')}")

if __name__ == '__main__':
    main()
```

**Test it:**
```bash
python data/fetch_cifar10.py --data-dir ./data
```

---

### 4. Single-GPU Trainer (2 hours)

Create `src/train.py` - baseline single-GPU version:
```python
"""
ResNet-18 training on CIFAR-10 (Single-GPU Baseline)
Usage: python src/train.py --epochs 5 --batch-size 128
"""
import argparse
import csv
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataloaders(data_dir, batch_size, num_workers=4):
    """Create CIFAR-10 train/test dataloaders"""
    # Standard CIFAR-10 augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=transform_test
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_time = time.time() - start_time
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    
    return avg_loss, accuracy, epoch_time

def validate(model, test_loader, criterion, device):
    """Evaluate on test set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    
    return avg_loss, accuracy

def save_metrics_csv(metrics, output_file):
    """Save metrics to CSV"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    file_exists = os.path.isfile(output_file)
    
    with open(output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 ResNet-18 Training')
    
    # Data
    parser.add_argument('--data', type=str, default='./data',
                        help='Path to CIFAR-10 dataset')
    
    # Training
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay')
    
    # System
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output
    parser.add_argument('--results', type=str, default='./results/csv',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get git commit hash (if in git repo)
    try:
        import subprocess
        git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except:
        git_hash = 'unknown'
    
    # Data
    train_loader, test_loader = get_dataloaders(
        args.data, args.batch_size, args.num_workers
    )
    
    # Model
    model = models.resnet18(num_classes=10)
    model = model.to(device)
    
    # Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}")
    print(f"Total batches per epoch: {len(train_loader)}")
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, epoch_time = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        scheduler.step()
        
        # Calculate throughput
        images_per_sec = len(train_loader.dataset) / epoch_time
        
        # Print progress
        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s | {images_per_sec:.0f} img/s")
        
        # Save metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'commit': git_hash,
            'epoch': epoch,
            'epochs': args.epochs,
            'world_size': 1,
            'gpus': 1,
            'batch_size': args.batch_size,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'epoch_time_s': epoch_time,
            'images_per_sec': images_per_sec,
        }
        
        output_file = os.path.join(args.results, 'baseline_1gpu.csv')
        save_metrics_csv(metrics, output_file)
    
    print(f"\nTraining complete! Results saved to {output_file}")

if __name__ == '__main__':
    main()
```

**Test locally (CPU or 1 GPU):**
```bash
python src/train.py --epochs 2 --batch-size 128 --data ./data
```

---

### 5. SYSTEM.md Documentation (30 minutes)

Create `docs/SYSTEM.md` (fill in YOUR cluster details):
```markdown
# System Configuration

## HPC Cluster Specifications

**Cluster Name:** [YOUR_CLUSTER_NAME]  
**Institution:** [YOUR_UNIVERSITY]  
**Access Date:** October 2025

### Compute Nodes
- **GPU Nodes:** X nodes available
- **GPUs per Node:** 4x [NVIDIA A100 / V100 / etc.]
- **GPU Memory:** 40GB / 80GB per GPU
- **CPUs per Node:** [e.g., 2x AMD EPYC 7742 (64 cores)]
- **RAM per Node:** [e.g., 512 GB]
- **Interconnect:** [InfiniBand HDR / Ethernet 100GbE]

### Software Stack
- **OS:** [e.g., Rocky Linux 8.8]
- **Slurm:** [version]
- **CUDA:** 12.1
- **cuDNN:** 8.9
- **NCCL:** 2.18
- **Python:** 3.10
- **PyTorch:** 2.0.1
- **torchvision:** 0.15.2

### Storage
- **Home Directory:** 50 GB quota
- **Scratch:** [path] (high-performance, temporary)
- **Project Space:** [path] (shared, persistent)

## Environment Setup

### Option Used: [Apptainer / Modules]

[Document your choice and specific setup steps]

## Job Submission

**Partition:** [gpu / gpu-dev / etc.]  
**Account:** [your_account]  
**QoS:** normal  
**Max Walltime:** 24 hours  
**Max Nodes:** 8 nodes

### Example sbatch
```bash
#SBATCH --partition=gpu
#SBATCH --account=<your_account>
#SBATCH --qos=normal
```

## Network Configuration

**NCCL Interface:** `export NCCL_SOCKET_IFNAME=ib0`  
**MPI:** OpenMPI 4.1.5

## Tested Configurations

| Date | Nodes | GPUs | Status | Notes |
|------|-------|------|--------|-------|
| 2025-10-17 | 1 | 1 | ✅ | Baseline working |
| TBD | 1 | 4 | ⏳ | DDP testing |
| TBD | 2 | 8 | ⏳ | Multi-node |

```

---

### 6. README.md (30 minutes)

Create `README.md`:
```markdown
# HPC CIFAR-10 Distributed Training

PyTorch DDP training of ResNet-18 on CIFAR-10 for HPC scaling study.

## 🎯 Project Goals
- Strong scaling efficiency ≥70% @ 8 GPUs
- Weak scaling: ≤10% time/epoch variance
- ≥20% throughput improvement via optimization
- Comprehensive profiling with Nsight Systems/Compute

## 📁 Repository Structure
```
├── src/          # Training code
├── env/          # Environment setup
├── slurm/        # Job scripts
├── data/         # Dataset utilities
├── results/      # Outputs (CSV, plots, logs)
└── docs/         # Documentation, paper, proposal
```

## 🚀 Quick Start

### 1. Setup
```bash
# Clone repo
git clone https://github.com/sakhnoukh/HPC.git
cd HPC

# Download data
python data/fetch_cifar10.py --data-dir ./data
```

### 2. Single-GPU Test
```bash
python src/train.py --epochs 5 --batch-size 128 --data ./data
```

### 3. Cluster Submission (TBD)
```bash
sbatch slurm/ddp_baseline.sbatch
```

## 📊 Current Status

**Week:** 1-2 (Foundation Setup)  
**Completed:**
- [x] Repository structure
- [x] Single-GPU baseline trainer
- [x] Data fetching script
- [ ] Environment (Apptainer/Modules)
- [ ] SYSTEM.md documentation
- [ ] DDP implementation

## 📚 Documentation
- [ROADMAP.md](ROADMAP.md) - Detailed project plan
- [STACK_AND_EXECUTION.md](STACK_AND_EXECUTION.md) - Tech stack & timeline
- [docs/SYSTEM.md](docs/SYSTEM.md) - Cluster specifications

## 👥 Team
[Add your team members]

## 📝 License
MIT

```

---

### 7. Test Everything (30 minutes)

```bash
# 1. Check directory structure
tree -L 2

# 2. Test data download
python data/fetch_cifar10.py --data-dir ./data

# 3. Test training (1 epoch, small batch)
python src/train.py --epochs 1 --batch-size 64 --data ./data

# 4. Verify CSV output
cat results/csv/baseline_1gpu.csv

# 5. Commit progress
git add .
git commit -m "Week 1: Initial setup with single-GPU baseline"
git push origin main
```

---

### 8. Transfer to Cluster (if working remotely)

```bash
# From local machine
rsync -avz --exclude 'data/' --exclude 'results/' \
  "/Users/samiakhnoukh/Documents/UNI/Year 3/Semester 1/HPC/HPC App/" \
  username@cluster:/path/to/HPC/

# On cluster
cd /path/to/HPC
python data/fetch_cifar10.py --data-dir ./data
python src/train.py --epochs 1 --batch-size 128 --data ./data
```

---

## 🎯 End of Week 1-2 Checklist

- [ ] Repository structure created
- [ ] Environment strategy chosen (Apptainer OR Modules)
- [ ] `data/fetch_cifar10.py` working
- [ ] `src/train.py` single-GPU baseline working
- [ ] CSV logging verified
- [ ] `docs/SYSTEM.md` documenting cluster specs
- [ ] `README.md` with project overview
- [ ] Code tested on cluster (at least 1 GPU)
- [ ] Git repo initialized and pushed to GitHub
- [ ] 200-word abstract drafted (optional)

---

## 🚧 Next Week Preview (Week 3-4)

**Goal:** Convert to DDP for multi-GPU on single node

**Key changes:**
1. Add `torch.distributed` initialization
2. Use `DistributedSampler`
3. Wrap model with `DDP()`
4. Create `slurm/ddp_baseline.sbatch`
5. Test 1 vs 2 vs 4 GPUs on single node

---

## ❓ Troubleshooting

### "CUDA out of memory"
→ Reduce `--batch-size` (try 64 or 32)

### "Cannot find CIFAR-10 dataset"
→ Run `python data/fetch_cifar10.py --data-dir ./data` first

### "Module not found: torchvision"
→ Install: `pip install torchvision` or check your environment setup

### Slow training
→ Check GPU usage: `nvidia-smi` (should be >80% utilization)  
→ Increase `--num-workers` (try 8)

---

**Good luck! 🚀**
```

---

## 🎬 Action Summary

**Today (Day 1):**
1. Create directory structure ✅
2. Choose environment strategy (Apptainer vs Modules)
3. Write `data/fetch_cifar10.py` and download data
4. Document cluster in `docs/SYSTEM.md`

**Tomorrow (Day 2):**
1. Implement `src/train.py` single-GPU
2. Test training for 1-2 epochs
3. Verify CSV logging
4. Write `README.md`

**By End of Week:**
1. Transfer code to cluster
2. Run full 5-epoch baseline on cluster GPU
3. Git commit and push to https://github.com/sakhnoukh/HPC
4. Draft 200-word abstract (optional)

---

**Questions to answer this week:**
- [ ] Which cluster are you using? (for SYSTEM.md)
- [ ] GPU model available? (A100/V100/etc.)
- [ ] Apptainer available on cluster, or modules only?
- [ ] Partition/account names for sbatch?

---

**Estimated Time:** 4-6 hours total for Week 1 tasks.
