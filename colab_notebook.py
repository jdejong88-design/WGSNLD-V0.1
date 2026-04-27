#!/usr/bin/env python3
"""
WatergeusLLM Colab Training Notebook (als .py voor reference)
Plak de cellen in Google Colab voor cloud training op A100

=== CELL 1: SETUP ===
"""

# Cell 1: Repo Clone + Dependencies
import os
os.chdir('/content')

# Clone repo
os.system("git clone https://github.com/jdejong88-design/WGSNLD-V0.1.git .")

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Install
os.system("pip install -q torch transformers datasets accelerate tokenizers tqdm")

print("[OK] Setup klaar!")

"""
=== CELL 2: VOORBEREIDING ===
"""

import torch
from pathlib import Path

# Paths
DRIVE_BASE = Path('/content/drive/MyDrive/WatergeusLLM')
LOCAL_BASE = Path('/content')

# Check GPU
print(f"[*] GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"[*] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Laad checkpoint (als aanwezig)
checkpoint_path = DRIVE_BASE / 'checkpoints' / 'checkpoint_batch_50000.pt'
if checkpoint_path.exists():
    print(f"[OK] Checkpoint gevonden: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    print(f"    Batch: {checkpoint.get('batch', '?')}")
    print(f"    Loss: {checkpoint.get('avg_loss', '?'):.4f}")
else:
    print("[!] Geen checkpoint gevonden — fresh start")

"""
=== CELL 3: TRAINING STARTEN ===
"""

os.chdir('/content')
os.system("python 04_train_model.py")

"""
=== CELL 4: RESULTATEN OPSLAAN ===
"""

import shutil

# Kopieer training history naar Drive
history_src = Path('/content/training_history.json')
history_dst = DRIVE_BASE / 'logs' / 'training_history.json'
if history_src.exists():
    history_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(history_src, history_dst)
    print(f"[OK] Training history opgeslagen naar Drive")

print("\n[OK] Training voltooid!")
