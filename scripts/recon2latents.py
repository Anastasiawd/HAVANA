#!/usr/bin/env python3
"""CLI wrapper for data_preparation/recon2latents.py."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_preparation.recon2latents import main  # noqa: E402


if __name__ == "__main__":
    main()
