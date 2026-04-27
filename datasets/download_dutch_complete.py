#!/usr/bin/env python3
"""Download Dutch datasets: CC100, Books, Oscar NL."""

import shutil
from pathlib import Path
from datasets import load_dataset

TARGET_DIR = Path("E:/Claude/workflow/WatergeusLLM/datasets")

DATASETS = [
    {
        "name": "CC100 Dutch",
        "repo_id": "cc100",
        "config": "nl",
        "output_dir": "cc100_nl",
        "split": "train",
        "desc": "Common Crawl 100 (2-3 GB)"
    },
    {
        "name": "Dutch Books Corpus",
        "repo_id": "nlp-databases/dutch-books",
        "config": None,
        "output_dir": "dutch_books",
        "split": "train",
        "desc": "Dutch Literature (1-2 GB)"
    },
    {
        "name": "Oscar NL Uncompressed",
        "repo_id": "oscar-corpus/OSCAR-2301",
        "config": "nl",
        "output_dir": "oscar_nl",
        "split": "train",
        "desc": "Web crawl text (8-10 GB)"
    }
]

def download_dataset(dataset_info):
    """Download and save a single dataset."""
    name = dataset_info["name"]
    repo_id = dataset_info["repo_id"]
    config = dataset_info["config"]
    output_dir = dataset_info["output_dir"]
    split = dataset_info["split"]
    desc = dataset_info["desc"]

    print(f"\n[*] {name} ({desc})...")
    print(f"    Repo: {repo_id}")

    try:
        # Load dataset
        if config:
            print(f"    Loading config: {config}...")
            ds = load_dataset(repo_id, config, split=split)
        else:
            print(f"    Loading...")
            ds = load_dataset(repo_id, split=split)

        num_rows = len(ds)
        print(f"[OK] Loaded {num_rows:,} rows")

        # Save to target
        target_path = TARGET_DIR / output_dir
        print(f"[*] Saving to {target_path}...")

        if target_path.exists():
            shutil.rmtree(target_path)

        ds.save_to_disk(str(target_path))
        print(f"[OK] Saved!")

        return True

    except Exception as e:
        error_msg = str(e)
        if "gated" in error_msg.lower():
            print(f"[ERROR] Dataset is gated. Accept it at https://huggingface.co/datasets/{repo_id}")
        else:
            print(f"[ERROR] {error_msg[:100]}")
        return False

def main():
    """Download all datasets."""
    print("=" * 60)
    print("[*] Dutch Datasets Download")
    print("=" * 60)
    print(f"Target: {TARGET_DIR}\n")

    results = {}
    for dataset_info in DATASETS:
        success = download_dataset(dataset_info)
        results[dataset_info["name"]] = success

    print("\n" + "=" * 60)
    print("[SUMMARY]")
    print("=" * 60)
    for name, success in results.items():
        status = "[OK]" if success else "[FAILED/GATED]"
        print(f"{status} {name}")

    print(f"\n[*] Final datasets in {TARGET_DIR.name}:")
    if TARGET_DIR.exists():
        items = sorted([d for d in TARGET_DIR.iterdir() if d.is_dir()])
        for item in items:
            print(f"    - {item.name}")

if __name__ == "__main__":
    main()
