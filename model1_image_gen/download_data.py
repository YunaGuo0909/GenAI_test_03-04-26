"""
Download animal image datasets for DDPM training.

Supports three datasets:
1. AFHQ v2 (Animal Faces HQ) — 3 classes, 15K images, 512x512 faces (~550 MB)
2. AWA2 (Animals with Attributes 2) — 50 classes, 37K images, varied sizes (~13 GB)
3. D&D (Dungeons & Diffusion) — 30 fantasy races, ~2.5K images (~200 MB) from HuggingFace

Usage:
    python model1_image_gen/download_data.py --dataset afhq
    python model1_image_gen/download_data.py --dataset dnd --data_dir /transfer
"""

import argparse
import json
import os
import struct
import sys
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).parent / "data"
DATA_DIR = DEFAULT_DATA_DIR

AFHQ_URL = "https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=1"
AWA2_URL = "https://cvml.ista.ac.at/AwA2/AwA2-data.zip"
DND_PARQUET_URL = "https://huggingface.co/datasets/0xJustin/Dungeons-and-Diffusion/resolve/main/data/train-00000-of-00001-6260e77b4303bc30.parquet"


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


def download_dnd():
    """Download Dungeons & Diffusion dataset from HuggingFace (fantasy character art)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dnd_dir = DATA_DIR / "dnd" / "images"

    if dnd_dir.exists() and len(list(dnd_dir.iterdir())) > 5:
        total = sum(len(list(d.glob("*"))) for d in dnd_dir.iterdir() if d.is_dir())
        print(f"D&D already exists at {dnd_dir} ({total} images)")
        return dnd_dir

    parquet_path = DATA_DIR / "dnd" / "dnd.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    if not parquet_path.exists():
        if not download_with_progress(DND_PARQUET_URL, str(parquet_path), "D&D parquet (~200 MB)"):
            print("\nTip: You can manually download from:")
            print("  https://huggingface.co/datasets/0xJustin/Dungeons-and-Diffusion")
            return None

    print("Extracting images from parquet (requires pyarrow or pandas)...")
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(str(parquet_path))
        df_mode = "pyarrow"
    except ImportError:
        try:
            import pandas as pd
            table = pd.read_parquet(str(parquet_path))
            df_mode = "pandas"
        except ImportError:
            print("  ERROR: Need pyarrow or pandas to read parquet files.")
            print("  Install with: pip install pyarrow")
            return None

    dnd_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    if df_mode == "pyarrow":
        n_rows = table.num_rows
        for i in range(n_rows):
            row = {col: table.column(col)[i].as_py() for col in table.column_names}
            img_data = row.get("image", {})
            label = row.get("text", "unknown")
            race = label.replace("D&D Character, ", "").split()[0].lower() if label else "unknown"

            race_dir = dnd_dir / race
            race_dir.mkdir(exist_ok=True)

            img_bytes = img_data.get("bytes") if isinstance(img_data, dict) else None
            if img_bytes:
                img_path = race_dir / f"{i:05d}.png"
                img_path.write_bytes(img_bytes)
                count += 1
            if (i + 1) % 500 == 0:
                print(f"  Extracted {i+1}/{n_rows} images...")
    else:
        n_rows = len(table)
        for i, row in table.iterrows():
            img_data = row.get("image", {})
            label = row.get("text", "unknown")
            race = label.replace("D&D Character, ", "").split()[0].lower() if label else "unknown"

            race_dir = dnd_dir / race
            race_dir.mkdir(exist_ok=True)

            img_bytes = img_data.get("bytes") if isinstance(img_data, dict) else None
            if img_bytes:
                img_path = race_dir / f"{i:05d}.png"
                img_path.write_bytes(img_bytes)
                count += 1
            if (count) % 500 == 0 and count > 0:
                print(f"  Extracted {count}/{n_rows} images...")

    classes = sorted(d.name for d in dnd_dir.iterdir() if d.is_dir())
    print(f"\nD&D ready! {count} images in {len(classes)} classes")
    for c in classes:
        n = len(list((dnd_dir / c).glob("*")))
        print(f"  {c}: {n}")
    return dnd_dir


def main():
    global DATA_DIR
    parser = argparse.ArgumentParser(description="Download animal image datasets")
    parser.add_argument("--dataset", type=str, default="afhq", choices=["afhq", "awa2", "dnd"],
                        help="Which dataset to download (default: afhq)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Custom directory to store datasets (default: model1_image_gen/data/)")
    args = parser.parse_args()

    if args.data_dir:
        DATA_DIR = Path(args.data_dir)

    if args.dataset == "afhq":
        path = download_afhq()
    elif args.dataset == "awa2":
        path = download_awa2()
    else:
        path = download_dnd()

    if path:
        print(f"\nDataset path for training: {path}")
        print(f"Use: python model1_image_gen/train_ddpm.py --data_root {path}")


if __name__ == "__main__":
    main()
