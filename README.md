# HPC CIFAR-10 Distributed Training

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

PyTorch Distributed Data Parallel (DDP) training of ResNet-18 on CIFAR-10 for HPC scaling study.

## 🎯 Project Goals

Train ResNet-18 on CIFAR-10 using PyTorch DDP under Slurm, scaling from 1 GPU to multiple GPU nodes. Deliver comprehensive performance analysis including:

- **Strong scaling efficiency** ≥70% at 8 GPUs (2 nodes)
- **Weak scaling** with ≤10% time/epoch variation
- **Optimization** showing ≥20% throughput improvement
- **Profiling** with Nsight Systems and Nsight Compute
- **Reproducible artifacts** (container/modules, CSV → plots)

## 📁 Repository Structure

```
HPC/
├── src/
│   ├── train.py              # Main DDP trainer
│   ├── data.py               # CIFAR-10 data loading
│   ├── utils.py              # Logging, seeding, metrics
│   └── plots/
│       └── make_all.py       # Generate figures from CSV
├── env/
│   ├── project.def           # Apptainer definition
│   ├── load_modules.sh       # Module loading script
│   └── requirements.txt      # Python dependencies
├── slurm/
│   ├── ddp_baseline.sbatch   # Single-node baseline
│   └── ...                   # Other sbatch scripts (TBD)
├── data/
│   ├── fetch_cifar10.py      # Dataset download script
│   └── README.md             # Dataset info
├── results/
│   ├── csv/                  # Experiment metrics
│   ├── plots/                # Generated figures
│   └── logs/                 # Slurm and profiling outputs
├── docs/
│   └── SYSTEM.md             # Cluster specifications
├── README.md                 # This file
├── ROADMAP.md                # Detailed project plan
├── STACK_AND_EXECUTION.md    # Technical stack & timeline
└── WEEK1_QUICKSTART.md       # Week 1-2 guide
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/sakhnoukh/HPC.git
cd HPC
```

### 2. Download CIFAR-10 Dataset
```bash
python data/fetch_cifar10.py --data-dir ./data
```

### 3. Setup Environment

**Option A: Local (pip)**
```bash
pip install -r env/requirements.txt
```

**Option B: Apptainer (on HPC cluster)**
```bash
apptainer build hpc_pytorch.sif env/project.def
```

**Option C: Modules (on HPC cluster)**
```bash
source env/load_modules.sh
```

### 4. Test Single-GPU Training
```bash
python src/train.py --epochs 2 --batch-size 128 --data ./data
```

### 5. Multi-GPU Training (DDP)
```bash
# 4 GPUs on single node
torchrun --nproc_per_node=4 src/train.py --epochs 5 --batch-size 128 --data ./data
```

### 6. HPC Cluster Submission (TBD)
```bash
sbatch slurm/ddp_baseline.sbatch
```

## 📊 Current Status

**Week:** 1-2 (Foundation Setup) ✅

**Completed:**
- [x] Repository structure
- [x] Single-GPU baseline trainer
- [x] DDP support in trainer
- [x] Data fetching script
- [x] Utility modules (logging, seeding, metrics)
- [x] Environment templates (Apptainer + Modules)
- [x] Initial documentation

**In Progress:**
- [ ] Slurm batch scripts
- [ ] Cluster testing (1 node, 4 GPUs)
- [ ] SYSTEM.md with cluster specs
- [ ] Multi-node scaling experiments

**Upcoming:**
- [ ] Strong/weak scaling analysis
- [ ] Profiling with Nsight
- [ ] Optimization (AMP, bucket tuning, dataloader)
- [ ] Paper and proposal writing

## 💻 Usage Examples

### Basic Training
```bash
# Single GPU, 10 epochs
python src/train.py --epochs 10 --batch-size 128 --data ./data --lr 0.1

# Specify output directory
python src/train.py --epochs 10 --results-dir ./results/csv --exp-name run1
```

### Distributed Training (torchrun)
```bash
# 2 GPUs
torchrun --nproc_per_node=2 src/train.py --epochs 10 --batch-size 128

# 4 GPUs
torchrun --nproc_per_node=4 src/train.py --epochs 10 --batch-size 128
```

### With Apptainer
```bash
apptainer exec --nv hpc_pytorch.sif \
    torchrun --nproc_per_node=4 src/train.py --epochs 5 --batch-size 128
```

## 🧪 Testing

Run a quick sanity check (1 epoch, small batch):
```bash
python src/train.py --epochs 1 --batch-size 64 --data ./data --print-freq 10
```

Expected output structure:
```
results/csv/baseline_1gpu_local.csv
```

CSV columns: timestamp, jobid, commit, epoch, world_size, gpus, train_loss, train_acc, val_acc, images_per_sec, etc.

## 📝 Key Features

- **Distributed Data Parallel (DDP)** - Multi-GPU/multi-node training
- **Automatic detection** - Single GPU or DDP based on environment
- **Reproducible** - Fixed seeds, deterministic operations
- **CSV logging** - All metrics saved for analysis
- **Flexible** - Many CLI arguments for experimentation
- **Optimized** - Pin memory, persistent workers, efficient dataloading

## 🔧 Configuration

Key command-line arguments:
- `--data`: Path to CIFAR-10 dataset directory
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size per GPU
- `--lr`: Learning rate (default: 0.1)
- `--num-workers`: Dataloader workers (default: 4)
- `--results-dir`: Output directory for CSV logs
- `--exp-name`: Experiment name for output files

See `python src/train.py --help` for all options.

## 📚 Documentation

- **[ROADMAP.md](ROADMAP.md)** - Complete 12-week project plan
- **[STACK_AND_EXECUTION.md](STACK_AND_EXECUTION.md)** - Technology stack and execution strategy
- **[WEEK1_QUICKSTART.md](WEEK1_QUICKSTART.md)** - Week 1-2 detailed guide
- **[docs/SYSTEM.md](docs/SYSTEM.md)** - HPC cluster specifications (TBD)

## 🤝 Contributing

This is a course project for HPC (High-Performance Computing). Team members:
- [Add team members here]

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- CIFAR-10 dataset: Alex Krizhevsky
- ResNet architecture: Kaiming He et al.
- PyTorch DDP: PyTorch Team
- HPC cluster: [Your institution]

## 📮 Contact

- **GitHub:** https://github.com/sakhnoukh/HPC
- **Issues:** https://github.com/sakhnoukh/HPC/issues

---

**Last Updated:** October 17, 2025  
**Status:** Week 1-2 - Foundation Complete ✅
