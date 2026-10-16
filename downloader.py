"""
Automatic dataset downloader for PRISM.
Attempts to download Roma Taxi and T-Drive if not already present.
"""

import os
import sys
import zipfile
import tarfile
import requests
from pathlib import Path
from typing import Optional
import subprocess


DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress."""
    try:
        print(f"[Download] {desc or url}")
        headers = {"User-Agent": "Mozilla/5.0 (compatible; PRISM-Research/1.0)"}
        with requests.get(url, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded * 100 // total
                            print(f"\r  {pct}% ({downloaded // 1024} KB)", end="")
            print()
        return True
    except Exception as e:
        print(f"[Download] Failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def setup_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials exist."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    print("\n[Kaggle] Credentials not found.")
    print("To enable automatic T-Drive download:")
    print("  1. Go to https://www.kaggle.com/settings -> API -> Create New Token")
    print("  2. Place kaggle.json in ~/.kaggle/")
    print("  3. chmod 600 ~/.kaggle/kaggle.json")
    return False


def download_tdrive_kaggle() -> bool:
    """Download T-Drive from Kaggle using the API."""
    dest_dir = DATA_DIR / "tdrive"
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print("[T-Drive] Already present.")
        return True

    if not setup_kaggle_credentials():
        return False

    try:
        import kaggle
        print("[T-Drive] Downloading from Kaggle (arashnic/tdriver)...")
        kaggle.api.dataset_download_files(
            "arashnic/tdriver",
            path=str(DATA_DIR),
            unzip=True,
            quiet=False,
        )
        # Kaggle usually creates a folder; move contents if needed
        extracted = DATA_DIR / "tdriver"  # or whatever the zip extracts to
        if not dest_dir.exists():
            # Try to find extracted folder
            for p in DATA_DIR.iterdir():
                if p.is_dir() and "tdrive" in p.name.lower():
                    p.rename(dest_dir)
                    break
        print("[T-Drive] Download complete.")
        return dest_dir.exists()
    except ImportError:
        print("[T-Drive] kaggle package not installed. Run: pip install kaggle")
        return False
    except Exception as e:
        print(f"[T-Drive] Kaggle download failed: {e}")
        return False


def download_roma_taxi() -> bool:
    """
    Attempt to download Roma Taxi.
    IEEE DataPort usually requires login, so we try known mirrors / direct links
    and fall back to instructions.
    """
    dest = DATA_DIR / "taxi_february.txt"
    if dest.exists() or (DATA_DIR / "roma_taxi.csv").exists():
        print("[Roma Taxi] Already present.")
        return True

    # Known public mirrors / direct attempts (these may break)
    candidate_urls = [
        # Add any known direct links here if discovered
        # Example placeholder (usually requires auth):
        # "https://ieee-dataport.org/open-access/download/...",
    ]

    for url in candidate_urls:
        tmp = DATA_DIR / "roma_temp.tar.gz"
        if download_file(url, tmp, "Roma Taxi"):
            try:
                if tarfile.is_tarfile(tmp):
                    with tarfile.open(tmp) as tar:
                        tar.extractall(DATA_DIR)
                elif zipfile.is_zipfile(tmp):
                    with zipfile.ZipFile(tmp) as z:
                        z.extractall(DATA_DIR)
                tmp.unlink()
                if dest.exists() or any(DATA_DIR.glob("*taxi*")):
                    print("[Roma Taxi] Extracted successfully.")
                    return True
            except Exception as e:
                print(f"[Roma] Extract failed: {e}")

    print("\n[Roma Taxi] Automatic download not possible (IEEE DataPort requires login).")
    print("Manual steps:")
    print("  1. Visit https://ieee-dataport.org/open-access/crawdad-romataxi")
    print("  2. Log in / create free account")
    print("  3. Download taxi_february.tar.gz")
    print("  4. Extract and place the .txt file in prism/data/")
    return False


def ensure_datasets(datasets: Optional[list] = None) -> dict:
    """
    Ensure requested datasets are available.
    Returns dict of {name: success}.
    """
    if datasets is None:
        datasets = ["tdrive", "roma"]

    results = {}
    print("=== Checking / Downloading Datasets ===\n")

    if "tdrive" in datasets or "t-drive" in datasets:
        results["tdrive"] = download_tdrive_kaggle()

    if "roma" in datasets or "roma_taxi" in datasets:
        results["roma"] = download_roma_taxi()

    print("\n=== Summary ===")
    for name, ok in results.items():
        status = "OK" if ok else "MANUAL DOWNLOAD REQUIRED"
        print(f"  {name}: {status}")

    return results


if __name__ == "__main__":
    ensure_datasets(["tdrive", "roma"])
