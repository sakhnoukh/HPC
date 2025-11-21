# HPC Assignment Completion Guide

**Last Updated:** November 21, 2025  
**Status:** Cluster experiments complete ✅ | Documentation in progress ⏳

---

## 📊 Current Status

### ✅ COMPLETED - Cluster Experiments

All experimental work on the Magic Castle cluster is **DONE**. We have:

1. **Baseline (1-GPU FP32)**
   - Throughput: 4,460 images/sec
   - Accuracy: 77.42%
   - Time/epoch: 11 seconds
   - Location: `results/csv/baseline_2644_1gpu_2644.csv`

2. **Multi-node Scaling (2-GPU)**
   - Throughput: 5,000 images/sec
   - Accuracy: 74.08%
   - Time/epoch: 20 seconds
   - Scaling efficiency: 56% (limited by network)
   - Location: `results/csv/baseline_2649_2gpu_2649.csv`

3. **Optimization (1-GPU FP16)**
   - Throughput: 7,600 images/sec
   - Accuracy: 78.7%
   - Time/epoch: 6.6 seconds
   - Speedup: **1.7x over FP32**
   - Location: `results/csv/fp16_2650_1gpu_2650.csv` (on cluster, needs download)

4. **Profiling**
   - GPU utilization: 90-95%
   - Bottleneck: Batch normalization (18% of compute time)
   - Memory: 455MB used (3% of 16GB)
   - Temperature: 39°C (healthy)
   - Location: `results/profiling/`

---

## ⏳ REMAINING WORK - Documentation (3-5 hours)

### Phase 1: Generate Plots (30 min)
### Phase 2: Write Short Paper (1.5 hours)
### Phase 3: Write EuroHPC Proposal (1.5 hours)
### Phase 4: Create Pitch Slides (30 min)
### Phase 5: Final Repository Cleanup (30 min)

---

## 📁 Repository Structure

```
HPC App/
├── src/
│   ├── train.py                  # Main DDP training script
│   ├── data.py                   # CIFAR-10 data loading
│   ├── utils.py                  # Helper functions
│   ├── train_slurm_wrapper.sh    # Maps Slurm vars to PyTorch DDP
│   ├── train_with_profiling.py   # PyTorch profiler script
│   └── analyze_profile.py        # Profile analysis script
│
├── slurm/
│   ├── ddp_baseline.sbatch       # 1-GPU FP32 baseline (USED)
│   ├── ddp_2gpu.sbatch           # 2-GPU multi-node (USED)
│   ├── ddp_fp16.sbatch           # 1-GPU FP16 optimization (USED)
│   ├── profile_pytorch.sbatch    # PyTorch profiling (USED)
│   └── test_*.sbatch             # Testing scripts
│
├── results/
│   ├── csv/                      # Performance metrics
│   │   ├── baseline_2644_1gpu_2644.csv      # 1-GPU FP32 ✅
│   │   ├── baseline_2649_2gpu_2649.csv      # 2-GPU multi-node ✅
│   │   └── fp16_2650_1gpu_2650.csv          # 1-GPU FP16 (on cluster)
│   │
│   ├── logs/                     # Slurm job logs
│   └── profiling/                # Profiling data
│       ├── gpu_util_2915.log     # GPU utilization
│       └── *.pt.trace.json       # PyTorch profiling trace
│
├── docs/
│   ├── SYSTEM.md                 # Cluster specs ✅
│   ├── paper.pdf                 # Short paper (TO DO)
│   ├── proposal.pdf              # EuroHPC proposal (TO DO)
│   └── slides.pdf                # 5-slide pitch (TO DO)
│
├── env/
│   └── project.def               # Apptainer container recipe
│
├── data/                         # CIFAR-10 dataset (auto-downloaded)
│
└── reproduce.md                  # Reproduction instructions (needs update)
```

---

## 🔧 PHASE 1: Generate Plots (30 min)

### What You Need to Do:

Create 3-4 plots from the CSV data to include in the paper.

### Required Plots:

1. **Throughput Comparison** (bar chart)
   - 1-GPU FP32: 4,460 img/s
   - 1-GPU FP16: 7,600 img/s
   - 2-GPU: 5,000 img/s

2. **Training Accuracy Curves** (line plot)
   - Show accuracy over epochs for each configuration

3. **Scaling Efficiency** (bar chart)
   - Ideal scaling (100%)
   - Actual 2-GPU scaling (56%)

4. **Profiling Breakdown** (pie/bar chart)
   - Batch norm: 18%
   - GEMM: 15.1%
   - Convolution: 13.7%
   - Other: 53.2%

### How to Generate Plots:

#### Option A: Using Existing Plot Script (Recommended)

Check if there's already a plotting script:
```bash
cd "HPC App"
ls -la src/plots/
```

If `make_all.py` exists, run it:
```bash
python3 src/plots/make_all.py
```

#### Option B: Create Simple Plotting Script

Create `src/plots/generate_figures.py`:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 1. Throughput Comparison
configs = ['1-GPU\nFP32', '1-GPU\nFP16', '2-GPU\nMulti-node']
throughputs = [4460, 7600, 5000]
colors = ['#3498db', '#2ecc71', '#e74c3c']

plt.figure()
plt.bar(configs, throughputs, color=colors)
plt.ylabel('Throughput (images/sec)')
plt.title('Training Throughput Comparison')
plt.ylim(0, 8000)
for i, v in enumerate(throughputs):
    plt.text(i, v + 200, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/throughput_comparison.png', dpi=300)
plt.close()

# 2. Accuracy Curves
df_1gpu = pd.read_csv('results/csv/baseline_2644_1gpu_2644.csv')
df_2gpu = pd.read_csv('results/csv/baseline_2649_2gpu_2649.csv')
# df_fp16 = pd.read_csv('results/csv/fp16_2650_1gpu_2650.csv')  # Download first

plt.figure()
plt.plot(df_1gpu['epoch'], df_1gpu['val_acc'], 'o-', label='1-GPU FP32', linewidth=2)
plt.plot(df_2gpu['epoch'], df_2gpu['val_acc'], 's-', label='2-GPU Multi-node', linewidth=2)
# plt.plot(df_fp16['epoch'], df_fp16['val_acc'], '^-', label='1-GPU FP16', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.title('Training Accuracy over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/accuracy_curves.png', dpi=300)
plt.close()

# 3. Scaling Efficiency
labels = ['Ideal\n(2x)', 'Actual\n(1.12x)']
values = [100, 56]
colors = ['#95a5a6', '#e74c3c']

plt.figure()
plt.bar(labels, values, color=colors)
plt.ylabel('Efficiency (%)')
plt.title('2-GPU Scaling Efficiency')
plt.ylim(0, 120)
plt.axhline(y=100, color='green', linestyle='--', label='Ideal')
for i, v in enumerate(values):
    plt.text(i, v + 3, f'{v}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/scaling_efficiency.png', dpi=300)
plt.close()

# 4. Profiling Breakdown
operations = ['Batch\nNorm', 'Matrix\nMultiply', 'Convolution', 'Other']
times = [18.0, 15.1, 13.7, 53.2]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#95a5a6']

plt.figure()
plt.bar(operations, times, color=colors)
plt.ylabel('Percentage of GPU Time (%)')
plt.title('GPU Operation Breakdown (Profiling)')
plt.ylim(0, 60)
for i, v in enumerate(times):
    plt.text(i, v + 1, f'{v}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/profiling_breakdown.png', dpi=300)
plt.close()

print("✓ All plots generated in results/plots/")
```

Run it:
```bash
mkdir -p results/plots
python3 src/plots/generate_figures.py
```

### Output:
- `results/plots/throughput_comparison.png`
- `results/plots/accuracy_curves.png`
- `results/plots/scaling_efficiency.png`
- `results/plots/profiling_breakdown.png`

---

## 📝 PHASE 2: Write Short Paper (1.5 hours)

### Assignment Requirements:
- **Length:** 4-6 pages
- **Format:** PDF
- **Location:** `docs/paper.pdf`

### Paper Outline:

```
1. Abstract (0.5 pages)
   - Problem: Training deep learning models faster
   - Approach: DDP on HPC cluster
   - Results: 1.7x speedup with FP16, 56% multi-node efficiency
   - Conclusion: Batch norm is bottleneck, network limits scaling

2. Introduction (0.5 pages)
   - Deep learning training is compute-intensive
   - Goal: Scale ResNet-18 training across GPUs
   - Research questions:
     * How does multi-node DDP scale?
     * What's the impact of mixed precision?
     * What are the bottlenecks?

3. Methodology (1 page)
   - System: Magic Castle cluster (2x Tesla T4 GPUs)
   - Software: PyTorch 2.0.1, Apptainer container
   - Model: ResNet-18 on CIFAR-10
   - Experiments:
     * Baseline: 1-GPU FP32
     * Scaling: 2-GPU multi-node
     * Optimization: 1-GPU FP16

4. Results (1.5 pages)
   - Include all 4 plots
   - Throughput: FP16 gives 1.7x speedup
   - Scaling: 2-GPU only 56% efficient (network bottleneck)
   - Profiling: Batch norm = 18% of compute time
   - Accuracy: All configs converge similarly (~75-78%)

5. Analysis (1 page)
   - Bottleneck analysis:
     * Batch norm overhead
     * Ethernet network limits multi-node
     * Small batch size underutilizes memory
   - Comparison to literature
   - Lessons learned

6. Conclusion & Future Work (0.5 pages)
   - FP16 is effective for single-GPU
   - Multi-node needs better interconnect (InfiniBand)
   - Future: Larger batch sizes, fused batch norm, more nodes
```

### Key Data to Include:

| Configuration | Throughput | Time/Epoch | Accuracy | Speedup |
|--------------|------------|------------|----------|---------|
| 1-GPU FP32 | 4,460 img/s | 11s | 77.42% | 1.0x |
| 1-GPU FP16 | 7,600 img/s | 6.6s | 78.7% | 1.7x |
| 2-GPU Multi-node | 5,000 img/s | 20s | 74.08% | 1.12x |

**Profiling Results:**
- GPU Utilization: 90-95%
- Memory Usage: 455MB (3% of 16GB)
- Bottleneck: Batch normalization (18% of compute)
- Temperature: 39°C (healthy)

### Templates:

You can use LaTeX or Word. For LaTeX, start with IEEE conference template:
```bash
# Download IEEE template
# Or use Overleaf: https://www.overleaf.com/latex/templates
```

---

## 🚀 PHASE 3: Write EuroHPC Proposal (1.5 hours)

### Assignment Requirements:
- **Length:** 6-8 pages (excluding references)
- **Format:** PDF
- **Location:** `docs/proposal.pdf`

### Proposal Outline:

```
1. Abstract & Objectives (1 page)
   - Propose scaling CIFAR-10 training to 16-32 GPUs
   - Goal: Study weak/strong scaling at larger scale
   - Request: 500 GPU-node-hours on EuroHPC system

2. State of the Art (1 page)
   - Distributed deep learning (Horovod, DDP)
   - Mixed precision training (NVIDIA Apex)
   - Related work on image classification scaling

3. Current Code & TRL (1.5 pages)
   - Technology Readiness Level: 5-6 (prototype validated)
   - What we've built:
     * PyTorch DDP implementation
     * Apptainer containerization
     * Reproducible experiments
   - Current results (copy from paper)
   - Code maturity: Ready for larger scale

4. Target EuroHPC Machine & Stack (1.5 pages)
   - Proposed system: LUMI-G (AMD MI250X GPUs)
     * Or MeluXina (NVIDIA A100)
   - Software stack:
     * PyTorch 2.0+
     * ROCm 5.x / CUDA 12.x
     * RCCL / NCCL for multi-GPU
     * Apptainer for reproducibility
   - Justification: InfiniBand for better scaling

5. Work Plan (1.5 pages)
   - Milestone 1 (Month 1): Port to target system
   - Milestone 2 (Month 2): Strong scaling (1-32 GPUs)
   - Milestone 3 (Month 3): Weak scaling experiments
   - Milestone 4 (Month 4): Optimize & document
   - Risks: Network bandwidth, batch size tuning
   - Support needed: System access, technical support

6. Resource Justification (1 page)
   - Calculation:
     * Strong scaling: 8 runs × 10 configs × 2 hours = 160 GPU-hours
     * Weak scaling: 8 runs × 8 configs × 3 hours = 192 GPU-hours
     * Optimization: 50 runs × 1 hour = 50 GPU-hours
     * Contingency: 100 GPU-hours
     * **Total: ~500 GPU-node-hours**
   - Formula: nodes × GPUs × hours × runs

7. Data Management & FAIR (0.5 pages)
   - Data: CIFAR-10 (public, open-source)
   - Results: CSV files, logs (GitHub)
   - FAIR principles: All code & data public
   - No ethical concerns (public dataset)

8. Expected Impact (0.5 pages)
   - Scientific: Understanding scaling limits
   - Educational: Training HPC students
   - Community: Open-source implementation
```

### EuroHPC Systems to Consider:

- **LUMI** (Finland): AMD MI250X GPUs, #3 in Top500
- **MeluXina** (Luxembourg): NVIDIA A100 GPUs
- **Leonardo** (Italy): NVIDIA A100 GPUs

Pick one based on availability/access.

---

## 🎤 PHASE 4: Create Pitch Slides (30 min)

### Assignment Requirements:
- **Length:** 5 slides
- **Time:** 5 minutes presentation
- **Format:** PDF
- **Location:** `docs/slides.pdf`

### Slide Structure:

**Slide 1: Problem & Impact**
- Title: "Scaling Deep Learning Training on HPC"
- Problem: Training is slow on single GPU
- Impact: Faster training = faster research
- Visual: Image of growing model sizes over time

**Slide 2: Approach & Prototype**
- System: Magic Castle (2× Tesla T4)
- Model: ResNet-18 on CIFAR-10
- Tech: PyTorch DDP + Apptainer
- Visual: Architecture diagram (1-GPU → 2-GPU)

**Slide 3: Results**
- Include 2-3 key plots:
  * Throughput comparison
  * Scaling efficiency
- Key numbers:
  * 1.7x speedup with FP16
  * 56% multi-node efficiency
- Visual: Your generated plots

**Slide 4: EuroHPC Target & Resource Ask**
- Proposed system: LUMI-G or MeluXina
- Goals: Scale to 16-32 GPUs
- Resource request: 500 GPU-node-hours
- Timeline: 4 months
- Visual: Timeline with milestones

**Slide 5: Risks, Milestones & Support**
- Milestones:
  * Month 1: Port to EuroHPC
  * Month 2-3: Scaling experiments
  * Month 4: Documentation
- Risks: Network bandwidth, batch tuning
- Support needed: System access, technical help
- Visual: Gantt chart or risk matrix

### Tools:
- PowerPoint / Keynote
- Google Slides
- LaTeX Beamer

---

## 🔧 PHASE 5: Final Repository Cleanup (30 min)

### Tasks:

1. **Update `reproduce.md`** with exact commands:
```bash
# Edit reproduce.md to include:
- How to build container
- How to submit each job
- Expected outputs
- How to generate plots
```

2. **Download missing FP16 results** from cluster:
```bash
cd "/Users/samiakhnoukh/Documents/UNI/Year 3/Semester 1/HPC"
rsync -avz \
  user58@login1.hpcie.labs.faculty.ie.edu:~/HPC_App/results/csv/fp16_2650_1gpu_2650.csv \
  "HPC App/results/csv/"
```

3. **Create release tag:**
```bash
cd "HPC App"
git add .
git commit -m "Final submission: paper, proposal, slides"
git tag -a v1.0 -m "Final submission for HPC assignment"
git push origin main --tags
```

4. **Verify all deliverables:**
```bash
# Check that everything is in place:
ls -lh docs/paper.pdf
ls -lh docs/proposal.pdf
ls -lh docs/slides.pdf
ls -lh results/plots/*.png
ls -lh results/csv/*.csv
```

---

## 📊 Quick Reference: Key Numbers

### Performance Metrics:
- **1-GPU FP32:** 4,460 img/s, 77.42% accuracy
- **1-GPU FP16:** 7,600 img/s, 78.7% accuracy, **1.7x speedup**
- **2-GPU:** 5,000 img/s, 74.08% accuracy, **56% efficiency**

### Profiling:
- **GPU Utilization:** 90-95%
- **Bottleneck:** Batch normalization (18%)
- **Memory:** 455MB / 16GB (3%)
- **Temperature:** 39°C

### System:
- **Cluster:** Magic Castle HPC, IE University
- **GPUs:** 2× NVIDIA Tesla T4 (16GB each)
- **CPU:** Intel Xeon Platinum 8473C
- **Network:** Ethernet (eth0)
- **Software:** PyTorch 2.0.1, CUDA 11.7, Apptainer

---

## 📧 Cluster Access

If you need to access the cluster:

```bash
# SSH into cluster
ssh user58@login1.hpcie.labs.faculty.ie.edu

# Navigate to project
cd ~/HPC_App

# Check job history
sacct -u user58 --format=JobID,JobName,State,Elapsed,MaxRSS

# Download files
rsync -avz user58@login1.hpcie.labs.faculty.ie.edu:~/HPC_App/results/ ./results/
```

---

## ❓ Common Questions

**Q: Do we need to run more experiments?**  
A: **No.** All required experiments are done. Focus on documentation.

**Q: What if plots don't look good?**  
A: Use the simple plotting script above. Matplotlib defaults are fine for academic papers.

**Q: How technical should the paper be?**  
A: Balance technical depth with readability. Assume reader knows ML basics but not HPC specifics.

**Q: Can we reuse text from README.md?**  
A: Yes, but rewrite in formal academic style. Add citations to PyTorch, NCCL, etc.

**Q: Where to find references?**  
A: Google Scholar:
- "PyTorch Distributed Data Parallel"
- "Mixed Precision Training NVIDIA"
- "Deep Learning Scaling"

---

## ✅ Final Checklist

- [ ] Phase 1: Generate 4 plots (30 min)
- [ ] Phase 2: Write short paper (1.5 hours)
- [ ] Phase 3: Write EuroHPC proposal (1.5 hours)
- [ ] Phase 4: Create 5-slide pitch (30 min)
- [ ] Phase 5: Final cleanup & tag (30 min)
- [ ] Place PDFs in `docs/`
- [ ] Commit & push to GitHub
- [ ] Create release tag `v1.0`

**Estimated Total Time:** 4-5 hours

---

## 🆘 Need Help?

If you get stuck:

1. Check existing files in `docs/` for templates
2. Review `SYSTEM.md` for cluster specs
3. Look at CSV files for exact numbers
4. Check profiling results in `results/profiling/`

**All the data you need is already collected. Just document it!**

---

Good luck! 🚀
