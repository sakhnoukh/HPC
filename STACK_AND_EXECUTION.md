# HPC Assignment: Technology Stack & Execution Plan

## 🎯 Project Overview
**Objective:** Build and scale a PyTorch DDP trainer for ResNet-18 on CIFAR-10, achieving:
- Strong scaling efficiency ≥70% at 2 nodes (8 GPUs)
- Weak scaling: ≤10% time/epoch variation
- ≥20% throughput improvement via optimization
- Comprehensive profiling with Nsight Systems/Compute

**GitHub Repo:** https://github.com/sakhnoukh/HPC

---

## 📚 Technology Stack

### Core Framework
- **PyTorch** (≥2.0.0) - Deep learning framework with native DDP support
- **torchvision** - For CIFAR-10 dataset and ResNet-18 model
- **Python** (3.9-3.11) - Programming language

### Distributed Training
- **PyTorch DDP** (Distributed Data Parallel) - Multi-GPU/multi-node training
- **NCCL** (NVIDIA Collective Communications Library) - GPU communication backend
- **torch.distributed.run** - Launch utility for distributed processes

### HPC Infrastructure
- **Slurm** - Workload manager for job scheduling
- **CUDA** (11.8+ or 12.x) - GPU computing platform
- **NVIDIA GPUs** - Target: A100/V100 or similar (4 GPUs/node minimum)

### Containerization/Environment
**Option A (Recommended): Apptainer/Singularity**
- Base: NVIDIA PyTorch container (nvcr.io/nvidia/pytorch:23.12-py3)
- Benefits: Reproducibility, portability, isolated dependencies

**Option B: Environment Modules**
- Module stack: GCC/Clang → CUDA → OpenMPI/MPICH → Python → PyTorch
- Requires careful version management

### Profiling & Analysis
- **Nsight Systems** - Timeline profiling, kernel execution, NCCL communication
- **Nsight Compute** - Detailed kernel-level metrics (FLOPs, bandwidth, occupancy)
- **sacct/scontrol** - Slurm job statistics
- **nvidia-smi** - GPU utilization monitoring

### Optimization Techniques
1. **Automatic Mixed Precision (AMP)** - BF16/FP16 for Tensor Core acceleration
2. **Gradient Bucketing** - Overlap communication with computation
3. **Optimized DataLoader** - `pin_memory=True`, `persistent_workers=True`, multi-worker prefetch

### Data & Visualization
- **CIFAR-10** - 60K images (50K train, 10K test), 10 classes
- **matplotlib/seaborn** - Plotting scaling curves
- **pandas** - CSV data processing

### Version Control & CI/CD
- **Git/GitHub** - Source control
- **Git LFS** (optional) - For large profiling outputs
- **Markdown** - Documentation

---

## 🚀 Execution Plan (Week-by-Week)

### **Week 1-2: Foundation Setup** ✅
**Goal:** Environment + baseline single-GPU training

**Tasks:**
1. ✅ Create repository structure (following ROADMAP layout)
2. Set up Apptainer definition OR module load script
3. Implement `src/train.py` - single-GPU ResNet-18 on CIFAR-10
   - CLI arguments (batch size, epochs, lr, optimizer)
   - CSV logging (metrics per epoch)
   - Reproducible seeding
4. Create `data/fetch_cifar10.py` - automatic dataset download
5. Test on 1 GPU locally or on single cluster node
6. Write 200-word abstract

**Deliverables:**
- `env/project.def` OR `env/load_modules.sh`
- `src/train.py` (single-GPU baseline)
- `data/fetch_cifar10.py`
- Initial `README.md` and `SYSTEM.md`

---

### **Week 3-4: DDP Implementation (1 Node)** 🔧
**Goal:** Multi-GPU training on single node (4 GPUs)

**Tasks:**
1. Convert `train.py` to use `torch.distributed`
   - Initialize process group
   - Use `DistributedSampler`
   - Wrap model with `DDP(model)`
   - Synchronize metrics across ranks
2. Create `slurm/ddp_baseline.sbatch`
   ```bash
   #SBATCH -N 1
   #SBATCH --gpus-per-node=4
   srun python -m torch.distributed.run --nproc_per_node=4 src/train.py
   ```
3. Verify correctness:
   - Compare accuracy: 1 GPU vs 4 GPUs (should converge similarly)
   - Check gradient synchronization
4. Implement CSV logging schema (jobid, nodes, GPUs, throughput, etc.)

**Deliverables:**
- `src/train.py` with DDP support
- `slurm/ddp_baseline.sbatch`
- First scaling plot: 1 vs 2 vs 4 GPUs (single node)
- `reproduce.md` (baseline section)

---

### **Week 5-6: Multi-Node Scaling** 🌐
**Goal:** Scale to 2-3 nodes (8-12 GPUs)

**Tasks:**
1. Create multi-node Slurm scripts:
   - `slurm/ddp_strong_scaling.sbatch` (fixed global batch=512)
   - `slurm/ddp_weak_scaling.sbatch` (fixed per-GPU batch=128)
2. Configure NCCL environment:
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_SOCKET_IFNAME=<cluster_interface>  # e.g., ib0
   ```
3. Run scaling experiments:
   - Strong: 1, 2, 4, 8, 12 GPUs
   - Weak: 1, 2, 4, 8, 12 GPUs
   - Record: throughput, time/epoch, GPU util
4. Calculate efficiencies:
   - Strong: E_s(N) = T(1) / (N × T(N))
   - Weak: E_w(N) = T(1) / T(N)

**Deliverables:**
- Multi-node sbatch scripts
- `results/csv/` with all runs
- Draft scaling plots (images/s vs GPUs, time/epoch vs GPUs)
- Efficiency curves

---

### **Week 7-8: Profiling & Bottleneck Analysis** 🔍
**Goal:** Identify performance bottlenecks

**Tasks:**
1. Create profiling wrappers:
   - `slurm/profile_gpu_nsys.sbatch` (Nsight Systems)
   - `slurm/profile_gpu_ncu.sbatch` (Nsight Compute)
2. Collect profiles for:
   - 1 node (4 GPUs)
   - 2 nodes (8 GPUs)
3. Analyze Nsight Systems timeline:
   - Compute vs communication overlap
   - DataLoader I/O time
   - NCCL all-reduce duration
4. Analyze Nsight Compute:
   - Kernel FLOPs achieved
   - Memory bandwidth utilization
   - Occupancy
5. Classify bottleneck:
   - **Compute-bound:** Low GPU util → optimize kernels/model
   - **Communication-bound:** NCCL dominant → overlap gradients
   - **I/O-bound:** DataLoader gaps → increase workers, pin memory

**Deliverables:**
- Nsight reports in `results/logs/`
- Bottleneck analysis document (2-3 pages)
- Screenshots of key timeline sections

---

### **Week 9-10: Optimization Implementation** ⚡
**Goal:** Implement ≥1 optimization with ≥20% improvement

**Primary Optimizations:**

1. **Automatic Mixed Precision (AMP)**
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   with autocast(dtype=torch.bfloat16):
       loss = model(inputs)
   ```
   - Use BF16 on Ampere+ GPUs (A100, H100)
   - Fallback to FP16 + loss scaling on older GPUs

2. **Gradient Bucketing Optimization**
   ```python
   model = DDP(model, bucket_cap_mb=50)  # Tune 25-100
   ```
   - Smaller buckets → earlier all-reduce start
   - Verify overlap in Nsight timeline

3. **DataLoader Improvements**
   ```python
   DataLoader(
       dataset, 
       num_workers=8,
       pin_memory=True,
       persistent_workers=True,
       prefetch_factor=2
   )
   ```

4. **Sensitivity Sweep**
   - Create job array testing combinations:
     - `bucket_cap_mb ∈ {25, 50, 100}`
     - `batch_size ∈ {64, 128, 256}`
     - `precision ∈ {fp32, fp16, bf16}`

**Tasks:**
1. Implement all optimizations in `train.py`
2. Add CLI flags: `--precision`, `--bucket-cap-mb`
3. Re-run strong/weak scaling with optimizations
4. Compare before/after metrics
5. Document improvement: throughput, time/epoch, GPU util

**Deliverables:**
- Optimized `train.py`
- `slurm/ddp_sensitivity.sbatch`
- Before/after comparison plots
- Optimization impact report (quantitative)

---

### **Week 11: Paper Writing** 📝
**Goal:** Complete 4-6 page technical paper

**Structure:**
1. **Introduction** (0.5p)
   - Multi-node training importance
   - CIFAR-10 as testbed
2. **Background & Related Work** (0.5p)
   - DDP architecture
   - Prior scaling studies
3. **Implementation** (1.5p)
   - Environment (Apptainer/modules)
   - DDP setup
   - AMP and optimizations
4. **Experiments** (0.5p)
   - Hardware: node specs, GPU type
   - Parameters: batch sizes, learning rate
5. **Results** (1.5p)
   - Scaling plots (strong/weak)
   - Efficiency tables
   - Optimization impact
6. **Profiling Analysis** (0.5p)
   - Nsight findings
   - Bottleneck classification
7. **Discussion & Limitations** (0.5p)
   - Small dataset/model
   - Extension ideas

**Tools:**
- LaTeX (Overleaf or local)
- Template: IEEE/ACM conference format
- References: BibTeX

**Deliverables:**
- `docs/paper.pdf`
- `docs/paper.tex` (source)

---

### **Week 11: EuroHPC Proposal** 🇪🇺
**Goal:** Complete 6-8 page Dev Access proposal

**Structure:**
1. **Abstract & Objectives** (1p)
2. **State of the Art** (1p)
3. **Current Code & TRL** (1p)
   - Existing DDP trainer
   - Containerization status
4. **Target Machine & Stack** (1p)
   - Specify EuroHPC system (LUMI, MeluXina, etc.)
   - GPU type (A100/H100/MI250X)
   - Software stack
5. **Work Plan** (2p)
   - Milestones: port, scale, profile, optimize
   - Timeline (Gantt chart)
6. **Resource Justification** (1p)
   - Node-hours calculation:
     ```
     3 nodes × 0.75h × 10 runs × 3 repeats = 67.5 node-hours
     + profiling overhead → request 80-100 node-hours
     ```
7. **Data Management & FAIR** (0.5p)
   - CIFAR-10 licensing
   - Artifact release (GitHub + Zenodo)
8. **Expected Impact** (0.5p)
   - Insights for community
   - Training efficiency best practices

**Deliverables:**
- `docs/proposal.pdf`

---

### **Week 12: Finalization & Presentation** 🎬
**Goal:** Release-ready artifacts

**Tasks:**
1. **Freeze Results**
   - Final experiment runs with n=3 repeats
   - Generate all plots via `src/plots/make_all.py`
2. **Create 5-Slide Pitch**
   - Problem & Impact
   - Approach (DDP + AMP)
   - Scaling Results (key plots)
   - EuroHPC Ask (resource table)
   - Timeline & Support Needed
3. **Finalize Documentation**
   - `README.md` - project overview, quick start
   - `SYSTEM.md` - hardware specs, software versions
   - `reproduce.md` - step-by-step reproduction
4. **Create Git Release Tag**
   ```bash
   git tag -a v1.0 -m "Final submission: HPC DDP scaling study"
   git push origin v1.0
   ```
5. **Archive Large Files**
   - Upload profiling outputs to external storage (Zenodo, OSF)
   - Link in README

**Deliverables:**
- `docs/slides.pdf`
- Release tag `v1.0`
- Complete repository with all artifacts

---

## 📊 Key Metrics to Track

### Performance Metrics
| Metric | Target | How to Measure |
|--------|--------|----------------|
| Strong Scaling Efficiency | ≥70% @ 8 GPUs | E_s = T(1)/(N×T(N)) |
| Weak Scaling Efficiency | ≤10% degradation | Compare time/epoch |
| Throughput | Baseline dependent | Images/second from CSV |
| Optimization Gain | ≥20% improvement | Before/after comparison |

### Profiling Metrics
- **GPU Utilization:** Target >80% (nvidia-smi)
- **Communication Overhead:** <20% of step time (Nsight)
- **DataLoader Idle Time:** <5% (Nsight)
- **Kernel Efficiency:** >50% peak FLOPs (Nsight Compute)

### Reproducibility Checklist
- ✅ All experiments use fixed seeds
- ✅ Git commit hash logged in CSV
- ✅ Environment versions documented (`SYSTEM.md`)
- ✅ Container/module definition committed
- ✅ All plots reproducible from CSV via script

---

## 🎯 Critical Success Factors

### Must-Have Features
1. **Reproducibility:** Anyone can run `reproduce.md` and get same results
2. **Scalability:** Code runs on 1-12 GPUs without modification
3. **Profiling:** Clear bottleneck identification with evidence
4. **Optimization:** Measurable improvement with explanation
5. **Documentation:** Clear, concise, complete

### Common Pitfalls to Avoid
❌ **Not setting seeds** → Non-reproducible results  
❌ **Forgetting DistributedSampler** → Data duplication across GPUs  
❌ **Wrong batch size math** → Global batch ≠ per-GPU batch × world_size  
❌ **NCCL misconfig** → Hangs or slow communication  
❌ **No checkpoint sync** → Multiple ranks overwriting files  
❌ **Ignoring I/O** → DataLoader becomes bottleneck at scale  

### Best Practices
✅ **Start small:** Test on 1 GPU → 2 GPUs → 4 GPUs before multi-node  
✅ **Log everything:** CSV + stdout + sacct + Nsight  
✅ **Version control:** Commit after each working milestone  
✅ **Test early:** Run short epochs (1-2) for initial validation  
✅ **Profile strategically:** Don't profile full runs (expensive)  
✅ **Document as you go:** Update README/SYSTEM.md incrementally  

---

## 📁 Repository Structure (To Build)

```
HPC/
├── README.md                 # Project overview, quick start
├── ROADMAP.md               # ✅ Detailed week-by-week plan
├── STACK_AND_EXECUTION.md   # ✅ This document
├── SYSTEM.md                # Hardware/software specifications (TODO)
├── reproduce.md             # Step-by-step reproduction guide (TODO)
│
├── src/
│   ├── train.py             # DDP trainer (TODO)
│   ├── data.py              # CIFAR-10 datamodule (TODO)
│   ├── utils.py             # Logging, seeding, timers (TODO)
│   └── plots/
│       └── make_all.py      # CSV → figures (TODO)
│
├── env/
│   ├── project.def          # Apptainer definition (TODO)
│   └── load_modules.sh      # Module load script (TODO)
│
├── slurm/
│   ├── ddp_baseline.sbatch           (TODO)
│   ├── ddp_strong_scaling.sbatch     (TODO)
│   ├── ddp_weak_scaling.sbatch       (TODO)
│   ├── ddp_sensitivity.sbatch        (TODO)
│   ├── profile_gpu_nsys.sbatch       (TODO)
│   └── profile_gpu_ncu.sbatch        (TODO)
│
├── data/
│   ├── README.md            # Dataset description
│   └── fetch_cifar10.py     # Download script (TODO)
│
├── results/
│   ├── csv/                 # Experiment metrics (generated)
│   ├── plots/               # Generated figures (generated)
│   └── logs/                # Slurm/Nsight outputs (generated)
│
└── docs/
    ├── paper.pdf            # Technical paper (Week 11)
    ├── proposal.pdf         # EuroHPC proposal (Week 11)
    └── slides.pdf           # 5-slide pitch (Week 12)
```

---

## 🛠️ Immediate Next Steps (This Week)

### Priority 1: Environment Setup
```bash
# 1. Clone/initialize repo
cd ~/HPC
git init
git remote add origin https://github.com/sakhnoukh/HPC.git

# 2. Create directory structure
mkdir -p src/plots env slurm data results/{csv,plots,logs} docs

# 3. Choose: Apptainer OR Modules
# Option A: Create env/project.def
# Option B: Create env/load_modules.sh

# 4. Test environment on cluster
```

### Priority 2: Single-GPU Baseline
```bash
# 1. Implement src/train.py (single-GPU)
#    - CIFAR-10 loading
#    - ResNet-18 from torchvision
#    - Training loop
#    - CSV logging

# 2. Test locally
python src/train.py --epochs 1 --batch-size 128

# 3. Create data/fetch_cifar10.py
```

### Priority 3: Documentation
```bash
# 1. Write SYSTEM.md
#    - Cluster name and specs
#    - GPU model and count per node
#    - CUDA/PyTorch/NCCL versions
#    - Network topology (InfiniBand/Ethernet)

# 2. Write initial README.md
#    - Project description
#    - Installation instructions
#    - Basic usage example

# 3. Draft 200-word abstract
```

---

## 📚 Useful Resources

### PyTorch DDP Documentation
- [DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Distributed Training Docs](https://pytorch.org/docs/stable/distributed.html)
- [AMP Guide](https://pytorch.org/docs/stable/amp.html)

### Profiling
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

### Slurm
- [Slurm Quick Start](https://slurm.schedmd.com/quickstart.html)
- [sbatch Documentation](https://slurm.schedmd.com/sbatch.html)

### EuroHPC
- [Dev Access Program](https://eurohpc-ju.europa.eu/access/development-access_en)
- [LUMI Docs](https://docs.lumi-supercomputer.eu/)

---

## ✅ Definition of Done

Project is complete when:
- ✅ Code runs on 1-12 GPUs without modification
- ✅ `reproduce.md` successfully recreates all results
- ✅ Strong/weak scaling plots with ≥70% efficiency
- ✅ One optimization showing ≥20% improvement
- ✅ Nsight Systems timeline + Nsight Compute report
- ✅ 4-6 page paper in `docs/paper.pdf`
- ✅ 6-8 page EuroHPC proposal in `docs/proposal.pdf`
- ✅ 5-slide pitch in `docs/slides.pdf`
- ✅ Git release tag `v1.0` created
- ✅ All code, data, and artifacts publicly accessible

---

**Last Updated:** 2025-10-17  
**Maintainer:** Sami Akhnoukh  
**GitHub:** https://github.com/sakhnoukh/HPC
