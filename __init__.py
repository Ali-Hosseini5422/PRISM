"""
Dataset loaders for PRISM with automatic download support.

Supports:
- T-Drive via kagglehub (arashnic/tdriver)
- Roma Taxi (IEEE DataPort - requires manual login, fallback provided)
- High-fidelity synthetic non-stationary traces
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import Dict, Optional
from pathlib import Path
import shutil


def try_download_tdrive(target_dir: Path) -> Optional[Path]:
    """
    Attempt to download T-Drive dataset using kagglehub.
    Requires: pip install kagglehub
    Authentication: Kaggle API token (see https://www.kaggle.com/settings/api)
    """
    try:
        import kagglehub
    except ImportError:
        print("[T-Drive] kagglehub not installed. Run: pip install kagglehub")
        return None

    target_dir.mkdir(parents=True, exist_ok=True)
    print("[T-Drive] Attempting download via kagglehub (arashnic/tdriver)...")

    try:
        # This downloads to kagglehub cache; we then copy/link to target_dir
        cache_path = kagglehub.dataset_download("arashnic/tdriver")
        print(f"[T-Drive] Downloaded to cache: {cache_path}")

        # Copy relevant files into our data directory
        cache = Path(cache_path)
        for item in cache.rglob("*"):
            if item.is_file() and item.suffix.lower() in {".txt", ".csv", ".zip"}:
                dest = target_dir / item.name
                if not dest.exists():
                    shutil.copy2(item, dest)
        print(f"[T-Drive] Files copied to {target_dir}")
        return target_dir
    except Exception as e:
        print(f"[T-Drive] Auto-download failed: {e}")
        print("  → You may need to authenticate:")
        print("     1. Get token from https://www.kaggle.com/settings/api")
        print("     2. Run: kagglehub.login()   or set KAGGLE_API_TOKEN env var")
        print("  → Or download manually and place files in data/tdrive/")
        return None


def try_download_roma(target_dir: Path) -> Optional[Path]:
    """
    Roma Taxi is hosted on IEEE DataPort and requires login.
    Automatic download is not possible without credentials.
    This function only checks for existing files and prints instructions.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    candidates = list(target_dir.glob("taxi_february*")) + list(target_dir.glob("roma*"))
    if candidates:
        print(f"[Roma] Found existing file: {candidates[0]}")
        return candidates[0]

    print("[Roma Taxi] Automatic download not available (IEEE DataPort requires login).")
    print("  Manual steps:")
    print("  1. Go to: https://ieee-dataport.org/open-access/crawdad-romataxi")
    print("  2. Create free IEEE account if needed")
    print("  3. Download taxi_february.tar.gz")
    print("  4. Extract and place the .txt/.csv file in:", target_dir)
    return None


class TrajectoryDataset:
    def __init__(self, root: str = "data", max_workers: int = 200):
        self.root = Path(root)
        self.root.mkdir(exist_ok=True)
        self.max_workers = max_workers


class SyntheticNonStationaryDataset(TrajectoryDataset):
    """High-fidelity synthetic with controlled drift (always available)."""

    def __init__(self, n_workers: int = 100, n_tasks: int = 40, drift_frequency: int = 50, seed: int = 0):
        super().__init__()
        self.n_workers = n_workers
        self.n_tasks = n_tasks
        self.drift_frequency = drift_frequency
        self.rng = np.random.RandomState(seed)
        self.positions = self.rng.randn(n_workers, 2) * 8.0
        self.task_positions = self.rng.randn(n_tasks, 2) * 8.0
        self.base_costs = self.rng.uniform(0.3, 0.8, n_workers)
        self.qualities = self.rng.uniform(0.6, 0.95, n_workers)
        self.skills = self.rng.randn(n_workers, 8)
        self.drift_level = 0

    def get_slot(self, t: int, strategic_fraction: float = 0.2) -> Dict[str, torch.Tensor]:
        if t > 0 and t % self.drift_frequency == 0:
            self.drift_level += 1
            self.positions += self.rng.randn(2) * 1.5
            self.base_costs = np.clip(self.base_costs + self.rng.randn(self.n_workers) * 0.08, 0.1, 1.0)

        self.positions += self.rng.randn(self.n_workers, 2) * 0.25

        w_feat = np.zeros((self.n_workers, 16), dtype=np.float32)
        w_feat[:, :2] = self.positions
        w_feat[:, 2:10] = self.skills
        w_feat[:, 10] = self.qualities
        w_feat[:, 11] = self.base_costs

        t_feat = np.zeros((self.n_tasks, 16), dtype=np.float32)
        t_feat[:, :2] = self.task_positions
        t_feat[:, 2:10] = self.rng.randn(self.n_tasks, 8).astype(np.float32)

        theta = self.base_costs.copy()
        reported = theta.copy()
        n_s = int(strategic_fraction * self.n_workers)
        if n_s:
            idx = self.rng.choice(self.n_workers, n_s, replace=False)
            reported[idx] *= self.rng.uniform(1.1, 1.25, n_s)

        return {
            "worker_feats": torch.from_numpy(w_feat),
            "task_feats": torch.from_numpy(t_feat),
            "theta": torch.from_numpy(reported.astype(np.float32)),
            "true_theta": torch.from_numpy(theta.astype(np.float32)),
            "quality": torch.from_numpy(self.qualities.astype(np.float32)),
            "loads": torch.from_numpy(np.clip(self.rng.exponential(0.3, self.n_workers), 0, 2).astype(np.float32)),
            "residual_capacities": torch.from_numpy(np.clip(2.0 - self.rng.rand(self.n_workers), 0.2, 3).astype(np.float32)),
            "task_values": torch.from_numpy(self.rng.uniform(1.0, 3.0, self.n_tasks).astype(np.float32)),
            "slot": t,
            "drift_level": self.drift_level,
        }


class RomaTaxiDataset(TrajectoryDataset):
    def __init__(self, auto_download: bool = True, sample_workers: int = 150):
        super().__init__()
        self.sample_workers = sample_workers
        self.df = None
        data_dir = self.root / "roma"

        if auto_download:
            try_download_roma(data_dir)

        # Look for files
        files = list(data_dir.glob("*.txt")) + list(data_dir.glob("*.csv")) + list(self.root.glob("taxi_february*"))
        if files:
            self._load(files[0])
        else:
            print("[Roma] Using synthetic fallback.")
            self.fallback = SyntheticNonStationaryDataset(n_workers=sample_workers, seed=1)

    def _load(self, path: Path):
        try:
            self.df = pd.read_csv(path, header=None, names=["id", "time", "lon", "lat"], nrows=400_000)
            self.df["id"] = self.df["id"].astype("category").cat.codes
            ids = self.df["id"].unique()[:self.sample_workers]
            self.df = self.df[self.df["id"].isin(ids)]
            print(f"[Roma] Loaded {len(ids)} taxis from {path.name}")
        except Exception as e:
            print(f"[Roma] Parse error: {e}")
            self.df = None

    def get_slot(self, t: int, strategic_fraction: float = 0.2) -> Dict[str, torch.Tensor]:
        if self.df is None or len(self.df) < 100:
            return self.fallback.get_slot(t, strategic_fraction)

        start = (t * 80) % max(1, len(self.df) - 80)
        window = self.df.iloc[start:start + 80]
        grouped = window.groupby("id")[["lon", "lat"]].mean()
        pos = grouped.values[:self.sample_workers]
        n = len(pos)

        w = np.zeros((n, 16), dtype=np.float32)
        w[:, :2] = pos
        n_t = min(30, n)
        t_idx = np.random.choice(n, n_t, replace=False)
        tf = np.zeros((n_t, 16), dtype=np.float32)
        tf[:, :2] = pos[t_idx]

        theta = np.random.uniform(0.4, 0.9, n).astype(np.float32)
        return {
            "worker_feats": torch.from_numpy(w),
            "task_feats": torch.from_numpy(tf),
            "theta": torch.from_numpy(theta),
            "true_theta": torch.from_numpy(theta.copy()),
            "quality": torch.from_numpy(np.random.uniform(0.65, 0.95, n).astype(np.float32)),
            "loads": torch.rand(n),
            "residual_capacities": torch.rand(n) + 0.5,
            "task_values": torch.rand(n_t) * 2 + 1,
            "slot": t,
            "drift_level": 0,
        }


class TDriveDataset(TrajectoryDataset):
    def __init__(self, auto_download: bool = True, sample_workers: int = 200):
        super().__init__()
        self.sample_workers = sample_workers
        self.df = None
        data_dir = self.root / "tdrive"

        if auto_download:
            try_download_tdrive(data_dir)

        files = list(data_dir.glob("*.txt")) + list(data_dir.glob("*.csv"))
        if files:
            self._load(files)
        else:
            print("[T-Drive] Using synthetic fallback.")
            self.fallback = SyntheticNonStationaryDataset(n_workers=sample_workers, seed=42)

    def _load(self, files):
        try:
            dfs = []
            for f in files[:30]:
                dfs.append(pd.read_csv(f, header=None, names=["id", "time", "lon", "lat"], nrows=8_000))
            self.df = pd.concat(dfs, ignore_index=True)
            self.df["id"] = self.df["id"].astype("category").cat.codes
            ids = self.df["id"].unique()[:self.sample_workers]
            self.df = self.df[self.df["id"].isin(ids)]
            print(f"[T-Drive] Loaded {len(ids)} taxis.")
        except Exception as e:
            print(f"[T-Drive] Parse error: {e}")
            self.df = None

    def get_slot(self, t: int, strategic_fraction: float = 0.2) -> Dict[str, torch.Tensor]:
        if self.df is None or len(self.df) < 50:
            return self.fallback.get_slot(t, strategic_fraction)

        start = (t * 120) % max(1, len(self.df) - 120)
        window = self.df.iloc[start:start + 120]
        grouped = window.groupby("id")[["lon", "lat"]].mean()
        pos = grouped.values[:self.sample_workers]
        n = len(pos)

        w = np.zeros((n, 16), dtype=np.float32)
        w[:, :2] = pos
        n_t = min(40, n)
        t_idx = np.random.choice(n, n_t, replace=False)
        tf = np.zeros((n_t, 16), dtype=np.float32)
        tf[:, :2] = pos[t_idx]

        theta = np.random.uniform(0.35, 0.85, n).astype(np.float32)
        return {
            "worker_feats": torch.from_numpy(w),
            "task_feats": torch.from_numpy(tf),
            "theta": torch.from_numpy(theta),
            "true_theta": torch.from_numpy(theta.copy()),
            "quality": torch.from_numpy(np.random.uniform(0.6, 0.95, n).astype(np.float32)),
            "loads": torch.rand(n) * 0.7,
            "residual_capacities": torch.rand(n) + 0.6,
            "task_values": torch.rand(n_t) * 2.2 + 0.9,
            "slot": t,
            "drift_level": 0,
        }


def get_dataset(name: str = "synthetic", auto_download: bool = True, **kwargs) -> TrajectoryDataset:
    name = name.lower()
    if name in ("synthetic", "synth"):
        return SyntheticNonStationaryDataset(**kwargs)
    elif name in ("roma", "roma_taxi"):
        return RomaTaxiDataset(auto_download=auto_download, **kwargs)
    elif name in ("tdrive", "t-drive", "beijing"):
        return TDriveDataset(auto_download=auto_download, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")


if __name__ == "__main__":
    print("Testing dataset factory...")
    ds = get_dataset("synthetic")
    print("Synthetic OK:", ds.get_slot(0)["worker_feats"].shape)

    print("\n--- T-Drive auto-download test ---")
    get_dataset("tdrive", auto_download=True)

    print("\n--- Roma status ---")
    get_dataset("roma", auto_download=True)
