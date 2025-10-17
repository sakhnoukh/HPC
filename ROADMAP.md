# Vision–HPC (CIFAR‑10 ResNet‑18 DDP) — Project Roadmap

> A concrete, week‑by‑week plan to build, scale, and profile a multi‑node image recognition trainer that fits the assignment rubrics exactly.

---

## 0) Scope & Objectives

**Goal:** Train ResNet‑18 on CIFAR‑10 using PyTorch **Distributed Data Parallel (DDP)** under **Slurm**, scaling from 1 GPU to 2–3 GPU nodes (4 GPUs/node). Deliver strong/weak scaling plots, profiling evidence (Nsight), one optimization with measurable gains, and reproducible artifacts (container/modules, CSV → plots).

**Success metrics (examples):**

* Strong‑scaling efficiency ≥ **70%** at 2 nodes (8 GPUs) from 1 node baseline.
* Weak‑scaling: time/epoch change ≤ **10%** when increasing GPUs with fixed per‑GPU batch.
* Throughput ≥ **X** images/s on 2 nodes (set X after baseline).
* Optimization (AMP + overlap): ≥ **20%** throughput improvement vs fp32 baseline.

---

## 1) Milestones (mapped to the course checkpoints)

**W1–W2 (Prep):**

* Choose CPU/GPU nodes available; confirm module versions; decide **Apptainer vs modules**.
* Draft 200‑word abstract and success metrics.
* Scaffold repo; tiny data sample; initial `train.py` (single‑GPU) that saves CSV metrics.

**W3 (Checkpoint: Topic + repo scaffold):**

* Repo layout complete; `env/` skeleton ready; CIFAR‑10 fetch script working.

**W4–W5 (Prototype v0 on 1 node):**

* Implement DDP on a single node (4 GPUs): correctness baseline, fixed seeds, logs.
* Container or module stack finalized; `reproduce.md` first draft.

**W6 (Checkpoint: 1‑node runs + env ready):**

* Baseline metrics + first plot (images/s vs GPUs on **one node**).

**W7–W8 (Multi‑node + first profiling):**

* Scale to 2–3 nodes; run strong + weak scaling grids.
* First Nsight Systems timeline + Nsight Compute kernel report.

**W9 (Checkpoint: ≥4 nodes/GPUs + first plots + profiling notes):**

* Deliver draft scaling plots; note bottlenecks (NCCL vs dataloader).

**W10 (Optimization & re‑measure):**

* Implement AMP + bucket overlap + dataloader fixes; re‑run selected grid.
* Start paper sections 1–5; start proposal outline.

**W11 (Checkpoint: Proposal draft):**

* Full draft of EuroHPC Dev Access proposal; resource table complete.

**W12 (Finalization):**

* Freeze results; finalize paper (4–6p), proposal (6–8p), 5 slides, release tag.

---

## 2) Team Roles

* **PM / Coordinator** — timeline, issues, integration; owns `README.md`, `SYSTEM.md`.
* **Env/Build Lead** — Apptainer or modules; deterministic seeds; `env/` scripts.
* **DDP/Training Lead** — `train.py` correctness, CLI, logs, checkpoints.
* **Slurm Lead** — sbatch templates, arrays, profiling wrappers, sacct logging.
* **Perf/Profiling Lead** — experiment matrix, Nsight/`perf`, results schema.
* **Plots & Analysis Lead** — CSV→figures, efficiency, Nsight screenshots annotated.
* **Writing/Proposal Lead** — paper/proposal structure, references, FAIR.

Assign backups for each.

---

## 3) Repo Layout (concrete)

```
src/
  train.py           # DDP trainer
  data.py            # CIFAR-10 datamodule (download, augment, loaders)
  utils.py           # logging, seeding, metrics, timers
  plots/make_all.py  # CSV→PNG/SVG
  profiling/
    nsys_marks.py    # NVTX ranges for key phases (optional)

env/
  project.def        # Apptainer (or modules.txt + load_modules.sh)
  load_modules.sh

slurm/
  ddp_baseline.sbatch
  ddp_strong_scaling.sbatch
  ddp_weak_scaling.sbatch
  ddp_sensitivity.sbatch
  profile_gpu_nsys.sbatch
  profile_gpu_ncu.sbatch

data/
  README.md          # tiny sample description
  fetch_cifar10.py   # scripted download

results/
  csv/
  plots/
  logs/              # sacct/scontrol/Nsight outputs

docs/
  README.md
  paper.pdf (later)
  proposal.pdf (later)
  slides.pdf (later)

README.md
SYSTEM.md
reproduce.md
```

---

## 4) Environment & Reproducibility

**Option A — Apptainer (recommended):** `env/project.def` based on NVIDIA PyTorch container (or CPU base if needed). Install PyTorch + torchvision matching cluster CUDA; log `nvidia-smi` and driver/runtime.

**Option B — Modules:** `env/modules.txt` with explicit versions; `env/load_modules.sh` to `module purge && module load ...` in order (compiler, CUDA, MPI, Python, PyTorch). Record versions in `SYSTEM.md`.

**Determinism:** set all seeds; enable `torch.backends.cudnn.benchmark=False`; document any nondeterministic ops if present.

---

## 5) Data Plan (CIFAR‑10)

* Keep **tiny sample** (e.g., 100 images) under `data/sample/` to test pipelines.
* `data/fetch_cifar10.py` downloads full dataset to a cache (outside repo if large).
* Augmentations: standard (random crop, flip); sensitivity: enable/disable RandAugment (optional).

---

## 6) Training Code — Key Features

* **CLI args:** dataset path, epochs, per‑GPU batch, global batch, optimizer, lr schedule, precision (`fp32|bf16|fp16`), `bucket_cap_mb`, workers, output dir.
* **Logging:** CSV row per epoch and per run; save git hash, job id, node/GPU counts, timings (dataloader/forward/backward/comm), best val accuracy.
* **DDP correctness:** one process per GPU; `DistributedSampler`; barrier at epoch end; save checkpoints only on rank 0.
* **AMP:** use Autocast + GradScaler (or native bf16 on Ampere/Hopper where stable).

---

## 7) Slurm Templates (fill account/paths)

**Baseline (single node, 4 GPUs)** — `slurm/ddp_baseline.sbatch`

```bash
#!/bin/bash
#SBATCH -J cifar-ddp-baseline
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -o results/logs/ddp_base_%j.out

module purge
source env/load_modules.sh # or: ./run.sh python ...
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO

srun --ntasks-per-node=4 --gpus-per-task=1 \
  python -m torch.distributed.run --nproc_per_node=4 --master_port=29500 \
  src/train.py --dataset cifar10 --data ./data --epochs 5 \
  --batch-size 128 --optimizer sgd --lr 0.1 --weight-decay 5e-4 \
  --precision bf16 --bucket-cap-mb 50 --results ./results/csv/${SLURM_JOB_ID}
```

**Strong scaling** — fixed global batch (e.g., 512) across #GPUs

```bash
# slurm/ddp_strong_scaling.sbatch
# vary nodes/GPU count via --ntasks and per-GPU batch to keep global batch constant
```

**Weak scaling** — fixed per‑GPU batch (e.g., 128)

```bash
# slurm/ddp_weak_scaling.sbatch
# keep per-GPU batch constant as GPUs increase
```

**Sensitivity sweep** — job array over bucket sizes or amp on/off

```bash
# slurm/ddp_sensitivity.sbatch
#SBATCH --array=0-5
# map array id → (bucket_cap_mb, precision) combinations
```

**Profiling wrappers**

```bash
# Nsight Systems timeline
nsys profile -o results/logs/nsys_%j --trace=cuda,nvtx,osrt \
  srun ... python -m torch.distributed.run ... src/train.py ...

# Nsight Compute kernel metrics
ncu --set full --target-processes all --export results/logs/ncu_%j \
  srun ... python -m torch.distributed.run ... src/train.py ...
```

---

## 8) Experiment Matrix (minimum to meet rubric)

**Baseline:**

* 1 node, GPUs={1,2,4}, epochs=5, per‑GPU batch=128, precision=bf16.

**Strong scaling (fixed global batch = 512):**

* GPUs = {1,2,4} on 1 node; then {8,12} across 2–3 nodes. Adjust per‑GPU batch so sum=512.

**Weak scaling (fixed per‑GPU batch = 128):**

* Same GPU counts as above; track time/epoch.

**Sensitivity (choose one):**

* `bucket_cap_mb ∈ {25,50,100}` (AMP=bf16). **Or** batch ∈ {64,128,256}.

**Optimization (implement & re‑test subset):**

* Enable AMP + tune bucket overlap + dataloader improvements; rerun strong/weak at 1, 2 nodes.

Repeat each point **n=3** if time allows for error bars.

---

## 9) Metrics & Logging (CSV schema)

`results/csv/run_*.csv` (one row per epoch; include a final summary row):

```
jobid,commit,ts,epochs,epoch,world_size,nodes,gpus_per_node,global_batch,per_gpu_batch,
train_top1,val_top1,images_per_sec,time_epoch_s,dl_time_s,fw_time_s,bw_time_s,comm_time_s,
mem_gb,util_gpu,eff_strong,eff_weak
```

**Efficiency formulas**

* Standard math
  [ E_s(N) = \frac{T(1)}{N,T(N)} \times 100% ]
  Raw LaTeX
  `E_s(N) = \frac{T(1)}{N\,T(N)} \times 100\%`

* Standard math
  [ E_w(N) = \frac{T(1)}{T(N)} \times 100% ]
  Raw LaTeX
  `E_w(N) = \frac{T(1)}{T(N)} \times 100\%`

**System logs to collect**

```bash
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,AveCPU,AveRSS,MaxRSS,AllocTRES%40,State \
  > results/logs/sacct_${SLURM_JOB_ID}.txt
scontrol show job $SLURM_JOB_ID > results/logs/scontrol_${SLURM_JOB_ID}.txt
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv \
  > results/logs/nvsmi_${SLURM_JOB_ID}.csv
```

---

## 10) Profiling Plan & Bottleneck Analysis

* **Nsight Systems:** Verify **one process per GPU**; measure overlap of backward compute vs NCCL all‑reduce; check host‑to‑device copy timing (input pipeline).
* **Nsight Compute:** Inspect main conv/GEMM kernels → achieved FLOPs, DRAM BW, occupancy, warp stalls; compare fp32 vs bf16.
* **NCCL Debug Logs:** enable `NCCL_DEBUG=INFO` to confirm rings/trees, link speed, collective durations.
* **Classify bottleneck:** comm‑bound (NCCL), I/O‑bound (dataloader), or compute‑bound. Tie chosen optimization to top bottleneck and quantify before/after.

---

## 11) Optimization Plan (one meaningful change)

1. **AMP + Larger Global Batch** — reduce step count per epoch; leverage Tensor Cores.
2. **Overlap Communication** — tune `--bucket-cap-mb` to start all‑reduce earlier; verify overlap in Nsight timeline.
3. **Input Pipeline Fixes** — `pin_memory=True`, `num_workers=8`, `persistent_workers=True`, prefetch
   (optional: stage CIFAR to node‑local scratch to reduce FS contention).

Document the delta: images/s, time/epoch, comm% of step, GPU util.

---

## 12) Plotting Plan (results/plots)

* Images/s vs #GPUs (strong scaling).
* Time/epoch vs #GPUs (weak scaling).
* Efficiency curves (strong & weak).
* Before/after optimization bars for throughput and comm% of step.
* (Optional) Kernel roofline: achieved FLOPs vs arithmetic intensity (Nsight Compute).

All figures generated by `src/plots/make_all.py` from `results/csv/*` with saved CLI for reproducibility.

---

## 13) Paper Outline (4–6 pages)

1. **Problem & Motivation** — why multi‑node training matters; CIFAR‑10 as a controlled testbed.
2. **Approach & Implementation** — DDP design, AMP, environment/packaging, Slurm usage.
3. **Data & Experiments** — grid definition, node types, parameters.
4. **Results** — strong/weak scaling plots, accuracy table.
5. **Profiling & Bottlenecks** — Nsight/`sacct` evidence; categorize bound regime.
6. **Optimization & Impact** — quantitative improvement and discussion.
7. **Limitations & Next Steps** — small dataset/model, extension ideas (TinyImageNet, SimCLR).

---

## 14) EuroHPC Proposal Outline (6–8 pages excl. refs)

* **Abstract & Objectives**, **State of the Art**, **Current Code & TRL** (DDP trainer, containerized).
* **Target Machine & Stack** — GPU nodes (A100/H100 or MI‑class), CUDA/HIP, NCCL/RCCL, PyTorch.
* **Work Plan & Milestones** — port, scale, profile, optimize.
* **Risks & Needed Support** — profiler access, storage, queue constraints.
* **Resource Justification** — node‑hours (see below).
* **Data/FAIR & Ethics** — CIFAR licensing, reproducibility, artifact release.
* **Expected Impact** — insights into comm/compute overlap and training efficiency.

**Node‑hours formula**

* Standard math
  [ \text{node‑hours} = \text{nodes} \times (\text{GPU}) \times \text{hours} \times \text{runs} ]
  Raw LaTeX
  `\text{node‑hours} = \text{nodes} \times (\text{GPU}) \times \text{hours} \times \text{runs}`

**Example ask:** 3 GPU nodes × 0.75 h/run × 10 runs = **22.5 GPU‑node‑hours** (plus repeats/profiling → justify to ~40–60).

---

## 15) Pitch (5 slides)

1. **Problem & Impact** — faster vision training; why scaling matters.
2. **Approach & Prototype** — DDP + AMP; container; Slurm runners.
3. **Scaling & Profiling** — key plots; Nsight timeline showing overlap improvement.
4. **EuroHPC Target & Resource Ask** — machine, stack, node‑hours table.
5. **Risks, Milestones & Support Needed** — profiler/time/storage; timeline graphic.

---

## 16) Risks & Mitigations

* **Queue delays** → short epochs, job arrays, run off‑peak; prioritize profiling on tiny subsets.
* **Dataset I/O contention** → pre‑cache CIFAR; node‑local scratch; increase dataloader workers.
* **NCCL topology issues** → set `NCCL_SOCKET_IFNAME`; ensure 1 process/GPU; check `NCCL_DEBUG=INFO`.
* **Accuracy regressions with AMP** → monitor val accuracy; use bf16 when possible; scale loss if fp16.
* **Version drift** → freeze PyTorch/torchvision/CUDA versions; log `pip freeze` or container hash.

---

## 17) Definition of Done (DoD)

* Reproducible runs on ≥2 nodes; `reproduce.md` executes all experiments and recreates plots.
* Strong & weak scaling plots with efficiencies; sensitivity sweep; **one optimization** with quantified gains.
* At least one Nsight Systems timeline + one Nsight Compute kernel report; brief bottleneck write‑up.
* Paper + EuroHPC proposal PDFs in `docs/`; 5‑slide pitch in `docs/`; release tag created.

---

## 18) Immediate Next Actions (this week)

1. Lock **Apptainer vs modules** and draft `env/` accordingly.
2. Implement **single‑node DDP** in `src/train.py` with CSV logging and seeding.
3. Create **Slurm baseline script** and verify one 4‑GPU run; push logs/CSV.
4. Write `reproduce.md` (baseline section) and `SYSTEM.md` (node types, drivers, modules).
5. Draft the 200‑word abstract using this plan + set initial success metrics.

> Keep every CLI, log, and CSV. Every figure must be reproducible from `results/csv` by `plots/make_all.py`. This is your grading safety net.
