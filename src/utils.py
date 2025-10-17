"""
Utility functions for logging, seeding, metrics, and timers.
"""
import csv
import os
import random
import subprocess
import time
from datetime import datetime

import numpy as np
import torch


def set_seed(seed=42):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make cudnn deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_hash():
    """
    Get current git commit hash.
    
    Returns:
        Git hash string or 'unknown' if not in a git repo
    """
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']
        ).decode('ascii').strip()
        return git_hash
    except:
        return 'unknown'


def get_slurm_job_id():
    """
    Get Slurm job ID from environment.
    
    Returns:
        Job ID string or 'local' if not running under Slurm
    """
    return os.environ.get('SLURM_JOB_ID', 'local')


class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self):
        self.times = []
        self.start_time = None
        
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        
    def stop(self):
        """Stop the timer and record elapsed time."""
        if self.start_time is None:
            raise RuntimeError("Timer was not started")
        elapsed = time.time() - self.start_time
        self.times.append(elapsed)
        self.start_time = None
        return elapsed
    
    def reset(self):
        """Reset all recorded times."""
        self.times = []
        self.start_time = None
    
    def mean(self):
        """Get mean of recorded times."""
        return np.mean(self.times) if self.times else 0.0
    
    def sum(self):
        """Get sum of recorded times."""
        return sum(self.times)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name='metric'):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions.
    
    Args:
        output: Model predictions (logits)
        target: Ground truth labels
        topk: Tuple of k values
        
    Returns:
        List of top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


class CSVLogger:
    """CSV logger for experiment metrics."""
    
    def __init__(self, filepath, fieldnames):
        """
        Initialize CSV logger.
        
        Args:
            filepath: Path to CSV file
            fieldnames: List of column names
        """
        self.filepath = filepath
        self.fieldnames = fieldnames
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Write header if file doesn't exist
        if not os.path.exists(filepath):
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
    
    def log(self, row_dict):
        """
        Log a row to CSV.
        
        Args:
            row_dict: Dictionary with keys matching fieldnames
        """
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row_dict)


def format_time(seconds):
    """
    Format seconds into human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def save_checkpoint(state, filepath):
    """
    Save training checkpoint.
    
    Args:
        state: Dictionary containing model state, optimizer state, etc.
        filepath: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load training checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        
    Returns:
        Dictionary with checkpoint info (epoch, best_acc, etc.)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint
