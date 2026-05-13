#!/usr/bin/env python3
"""
scripts/setup_and_run.py
------------------------
One-shot setup: installs dependencies, verifies the environment,
then launches training + evaluation.

Run with:
    python scripts/setup_and_run.py
"""

import subprocess
import sys
from pathlib import Path


def run(cmd: str, check: bool = True) -> int:
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        sys.exit(f"Command failed (exit {result.returncode}): {cmd}")
    return result.returncode


def main():
    root = Path(__file__).parent.parent
    print("=" * 60)
    print("  Service Desk Ticket Classifier – Setup & Run")
    print("=" * 60)

    print("\n[1/3] Installing Python dependencies …")
    run(f"{sys.executable} -m pip install -r {root / 'requirements.txt'} --quiet")

    print("\n[2/3] Verifying PyTorch …")
    run(f"{sys.executable} -c \"import torch; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())\"")

    print("\n[3/3] Running training pipeline …")
    run(f"{sys.executable} {root / 'train.py'}")

    print("\n" + "=" * 60)
    print("  ✔  Setup complete!")
    print("  Run predictions with:")
    print('  python predict.py --text "My laptop screen is black"')
    print("=" * 60)


if __name__ == "__main__":
    main()
