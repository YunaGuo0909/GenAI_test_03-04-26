"""
Download animal image datasets for DDPM training.

Supports two datasets:
1. AFHQ v2 (Animal Faces HQ) — 3 classes, 15K images, 512x512 faces (~550 MB)
2. AWA2 (Animals with Attributes 2) — 50 classes, 37K images, varied sizes (~13 GB)

Both are organized as ImageFolder-compatible directory structures.

Usage:
    python model1_image_gen/download_data.py --dataset afhq
    python model1_image_gen/download_data.py --dataset awa2
"""

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

AFHQ_URL = "https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=1"
AWA2_URL = "https://cvml.ista.ac.at/AwA2/AwA2-data.zip"


def download_with_progress(url: str, dest: str, desc: str = ""):
    """Download a file with a progress indicator."""
    print(f"Downloading {desc}...")
    print(f"  URL: {url}")
    print(f"  Saving to: {dest}")

    def reporthook(count, block_size, total_size):
        percent = min(count * block_size * 100 // total_size, 100) if total_size > 0 else 0
        mb_done = count * block_size / (1024 * 1024)
        mb_total = total_size / (1024 * 1024) if total_size > 0 else 0
        sys.stdout.write(f"\r  [{percent:3d}%] {mb_done:.1f}/{mb_total:.1f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest, reporthook)
        print("\n  Download complete!")
        return True
    except Exception as e:
        print(f"\n  Download failed: {e}")
        return False


def download_afhq():
    """Download AFHQ v2 dataset (3 classes: cat, dog, wildlife)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    afhq_dir = DATA_DIR / "afhq"

    if afhq_dir.exists() and any(afhq_dir.iterdir()):
        print(f"AFHQ already exists at {afhq_dir}")
        return afhq_dir / "train"

    zip_path = DATA_DIR / "afhq_v2.zip"
    if not zip_path.exists():
        if not download_with_progress(AFHQ_URL, str(zip_path), "AFHQ v2 (~550 MB)"):
            return None

    print("Extracting AFHQ...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(DATA_DIR))

    for split in ["train", "val"]:
        split_dir = afhq_dir / split
        if split_dir.exists():
            classes = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
            counts = {c: len(list((split_dir / c).glob("*"))) for c in classes}
            print(f"  {split}: {counts}")

    print("AFHQ ready!")
    return afhq_dir / "train"


def download_awa2():
    """Download AWA2 dataset (50 animal classes)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    awa2_dir = DATA_DIR / "awa2"
    images_dir = awa2_dir / "JPEGImages"

    if images_dir.exists() and len(list(images_dir.iterdir())) > 10:
        print(f"AWA2 already exists at {images_dir}")
        return images_dir

    zip_path = DATA_DIR / "AwA2-data.zip"
    if not zip_path.exists():
        print("AWA2 is ~13 GB. Make sure you have enough disk space.")
        if not download_with_progress(AWA2_URL, str(zip_path), "AWA2 (~13 GB)"):
            print("\nTip: You can manually download AWA2 from:")
            print(f"  {AWA2_URL}")
            print(f"  Extract to: {awa2_dir}/")
            return None

    print("Extracting AWA2 (this may take a while)...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(DATA_DIR))

    # AWA2 extracts with nested structure, find the JPEGImages folder
    for candidate in [images_dir, awa2_dir / "Animals_with_Attributes2" / "JPEGImages"]:
        if candidate.exists():
            images_dir = candidate
            break

    if images_dir.exists():
        classes = sorted(d.name for d in images_dir.iterdir() if d.is_dir())
        print(f"  {len(classes)} animal classes found")
        total = sum(len(list((images_dir / c).glob("*"))) for c in classes[:5])
        print(f"  Sample counts (first 5): ~{total} images")

    print("AWA2 ready!")
    return images_dir


def main():
    parser = argparse.ArgumentParser(description="Download animal image datasets")
    parser.add_argument("--dataset", type=str, default="afhq", choices=["afhq", "awa2"],
                        help="Which dataset to download (default: afhq)")
    args = parser.parse_args()

    if args.dataset == "afhq":
        path = download_afhq()
    else:
        path = download_awa2()

    if path:
        print(f"\nDataset path for training: {path}")
        print(f"Use: python model1_image_gen/train_ddpm.py --data_root {path}")


if __name__ == "__main__":
    main()
