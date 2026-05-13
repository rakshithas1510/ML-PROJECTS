"""
src/utils.py
------------
Shared utilities: config loading, logging setup, reproducibility seeding.
"""

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


# ─── Config ──────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file and return as a nested dict."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


# ─── Logging ─────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO", log_file: str = None) -> None:
    """
    Configure root logger with console + optional file handler.

    Parameters
    ----------
    level    : logging level string, e.g. "INFO", "DEBUG"
    log_file : path to log file (None = console only)
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level   = getattr(logging, level.upper(), logging.INFO),
        format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        handlers= handlers,
        force   = True,
    )


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility across Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─── Device ──────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available device (CUDA -> MPS -> CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Device] Using: {device}")
    return device


# ─── Results Persistence ─────────────────────────────────────────────────────

def save_results(results: dict, path: str) -> None:
    """Serialise a results dict (without numpy arrays) to JSON."""
    serialisable = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            serialisable[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            serialisable[k] = v.item()
        else:
            serialisable[k] = v

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    logging.getLogger(__name__).info("Results saved -> %s", path)
