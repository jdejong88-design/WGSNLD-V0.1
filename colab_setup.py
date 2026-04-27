#!/usr/bin/env python3
"""
Colab Setup Script voor WatergeusLLM
Voer dit in Colab uit (Cell 1) voor volledige cloud-integratie
"""

import os
import shutil
from pathlib import Path

def setup_colab():
    """Automatische setup voor Google Colab"""

    print("=" * 70)
    print("[*] WatergeusLLM Colab Setup")
    print("=" * 70)

    # Stap 1: Clone repository
    print("\n[1] GitHub Repository Clonen")
    print("-" * 50)
    os.system("git clone https://github.com/jdejong88-design/WGSNLD-V0.1.git /content/repo")
    print("[OK] Repository gecloned naar /content/repo")

    # Stap 2: Mount Google Drive
    print("\n[2] Google Drive Mounten")
    print("-" * 50)
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("[OK] Google Drive gemount op /content/drive")
    except ImportError:
        print("[!] Niet in Colab — mount Drive handmatig")

    # Stap 3: Dependencies installeren
    print("\n[3] Dependencies Installeren")
    print("-" * 50)
    os.system("pip install -q torch transformers datasets accelerate tokenizers tqdm")
    print("[OK] Dependencies geïnstalleerd")

    # Stap 4: Folder-structuur in Drive maken
    print("\n[4] Google Drive Folders Voorbereiden")
    print("-" * 50)
    drive_base = Path('/content/drive/MyDrive/WatergeusLLM')
    drive_base.mkdir(parents=True, exist_ok=True)
    (drive_base / 'checkpoints').mkdir(exist_ok=True)
    (drive_base / 'tokens').mkdir(exist_ok=True)
    (drive_base / 'logs').mkdir(exist_ok=True)
    print(f"[OK] Folders aangemaakt in {drive_base}")

    # Stap 5: Tokenizer kopiëren (van locale schijf of repo)
    print("\n[5] Tokenizer Klaarzetten")
    print("-" * 50)
    tokenizer_src = Path('/content/repo/dutch_tokenizer.json')
    tokenizer_dst = drive_base / 'dutch_tokenizer.json'
    if tokenizer_src.exists():
        shutil.copy(tokenizer_src, tokenizer_dst)
        print(f"[OK] Tokenizer gekopieerd naar Drive")
    else:
        print("[!] Tokenizer niet gevonden — upload handmatig naar Drive")

    # Stap 6: Checkpoint laden (als aanwezig in Drive)
    print("\n[6] Checkpoints Detecteren")
    print("-" * 50)
    checkpoints_dir = drive_base / 'checkpoints'
    checkpoints = list(checkpoints_dir.glob('checkpoint_batch_*.pt'))
    if checkpoints:
        latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        print(f"[OK] Gevonden: {latest.name}")
        print(f"    Hervatting training van batch {latest.stem.split('_')[-1]}")
    else:
        print("[!] Geen checkpoints gevonden — fris begin")

    print("\n" + "=" * 70)
    print("[OK] Colab Setup Voltooid!")
    print("=" * 70)
    print("\nVolgende stap: python /content/repo/04_train_model.py")

if __name__ == "__main__":
    setup_colab()
