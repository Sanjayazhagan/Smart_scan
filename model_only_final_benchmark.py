"""Canonical model-only benchmark for the official Turing stare test split.

Run from the repository root:
    python model_only_final_benchmark.py

The runner evaluates every official ``stare/test_stare/*.h5`` mission with the
same settings and prints the aggregate leaderboard sorted by mean reward. It
never selects a mission to make Smart Scan look better.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from model_only_stare_full_benchmark import main


if __name__ == "__main__":
    main()
