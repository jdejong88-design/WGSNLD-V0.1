# Google Colab Setup Guide — WatergeusLLM

**Doel:** Training hernemen op A100 GPU voor 100x snelheid boost.

---

## Fase 1: Google Drive Voorbereiding (VOOR Colab)

### Stap 1.1: Maak folder-structuur aan

In Google Drive, maak deze structuur:
```
MyDrive/
└── WatergeusLLM/
    ├── checkpoints/          ← Checkpoint bestanden
    ├── tokens/               ← Tokenized data (.pt bestanden)
    ├── logs/                 ← Training logs & history
    └── dutch_tokenizer.json  ← Tokenizer bestand
```

### Stap 1.2: Upload checkpoint_batch_50000.pt

**Lokale locatie:** `E:/Claude/workflow/WatergeusLLM/checkpoints/checkpoint_batch_50000.pt`

**Drive bestemming:** `MyDrive/WatergeusLLM/checkpoints/checkpoint_batch_50000.pt`

**File size:** ~141 MB  
**Upload time:** ~5-10 minuten

### Stap 1.3: Upload tokenized data (tokens/)

**Lokale locatie:** `E:/Claude/workflow/WatergeusLLM/tokens/` (alle `.pt` bestanden)

**Drive bestemming:** `MyDrive/WatergeusLLM/tokens/`

**Voorwaarde:** `02_tokenization.py` moet lokaal gedraaid zijn!

**File size:** ~500 MB+  
**Upload time:** ~1-2 uur (zip eerst!)

### Stap 1.4: Upload tokenizer

**Lokale locatie:** `E:/Claude/workflow/WatergeusLLM/dutch_tokenizer.json`

**Drive bestemming:** `MyDrive/WatergeusLLM/dutch_tokenizer.json`

**File size:** <1 MB  
**Upload time:** Instant

---

## Fase 2: GitHub Voorbereiding

### Stap 2.1: Verifieer GitHub repo

Open: https://github.com/jdejong88-design/WGSNLD-V0.1

Controleer:
- ✅ `03_build_transformer_model.py` aanwezig
- ✅ `04_train_model.py` aanwezig
- ✅ `04_train_model_optimized.py` aanwezig (voor FP16)
- ✅ `colab_setup.py` aanwezig
- ✅ `.gitignore` negeert `checkpoints/`, `tokens/`, `*.log`

### Stap 2.2: SSH Key voor Colab (OPTIONEEL)

Als HTTPS auth fails, use SSH:

```bash
# Lokaal: Genereer SSH key
ssh-keygen -t ed25519 -C "jdejong88@gmail.com"

# Kopieer public key naar GitHub Settings > SSH Keys
```

In Colab:
```python
# Zet private key in Colab secrets
from google.colab import userdata
ssh_key = userdata.get('GITHUB_SSH_PRIVATE_KEY')
```

---

## Fase 3: Colab Notebook Setup

### Cell 1: Environment Setup

```python
# Install dependencies
!pip install -q torch transformers datasets accelerate tokenizers tqdm

# Clone repo
!git clone https://github.com/jdejong88-design/WGSNLD-V0.1.git /content/repo
%cd /content/repo

# Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Cell 2: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Verify folder structure
import os
drive_path = '/content/drive/MyDrive/WatergeusLLM'
print(f"Drive path exists: {os.path.exists(drive_path)}")
print(f"Subfolders: {os.listdir(drive_path)}")
```

### Cell 3: Load Checkpoint (Optional)

```python
import torch
from pathlib import Path

checkpoint_path = Path('/content/drive/MyDrive/WatergeusLLM/checkpoints/checkpoint_batch_50000.pt')

if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    print(f"✅ Checkpoint loaded!")
    print(f"   Batch: {checkpoint.get('batch')}")
    print(f"   Loss: {checkpoint.get('avg_loss'):.4f}")
else:
    print("⚠️  No checkpoint found — fresh start")
```

### Cell 4: Setup Data Paths

```python
# Link tokenizer
import shutil
tokenizer_src = Path('/content/drive/MyDrive/WatergeusLLM/dutch_tokenizer.json')
tokenizer_dst = Path('/content/repo/dutch_tokenizer.json')

if tokenizer_src.exists() and not tokenizer_dst.exists():
    shutil.copy(tokenizer_src, tokenizer_dst)
    print(f"✅ Tokenizer copied to repo")

# Link tokens
tokens_dst = Path('/content/repo/tokens')
if not tokens_dst.exists():
    tokens_dst.symlink_to(Path('/content/drive/MyDrive/WatergeusLLM/tokens'))
    print(f"✅ Tokens symlinked")

# Link checkpoints
checkpoints_dst = Path('/content/repo/checkpoints')
if not checkpoints_dst.exists():
    checkpoints_dst.mkdir()
    print(f"✅ Checkpoints folder created")
```

### Cell 5: Start Training (FP32)

```python
# Standard training
!python 04_train_model.py
```

### Cell 5b: Start Training (FP16 RECOMMENDED)

```python
# Optimized with mixed precision
!python 04_train_model_optimized.py
```

### Cell 6: Save Results

```python
import shutil
from pathlib import Path

# Copy training history back to Drive
history_src = Path('/content/repo/training_history.json')
history_dst = Path('/content/drive/MyDrive/WatergeusLLM/logs/training_history.json')

if history_src.exists():
    history_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(history_src, history_dst)
    print(f"✅ Training history saved to Drive")

# Copy best model
best_model_src = Path('/content/repo/checkpoints/best_model.pt')
best_model_dst = Path('/content/drive/MyDrive/WatergeusLLM/checkpoints/best_model.pt')

if best_model_src.exists():
    shutil.copy(best_model_src, best_model_dst)
    print(f"✅ Best model saved to Drive")
```

---

## Fase 4: Performance Tips

### Mixed Precision (FP16)
- Use `04_train_model_optimized.py` instead of `04_train_model.py`
- ~50% geheugen besparing
- 20-40% sneller
- A100 ondersteunt dit perfect

### Batch Size Tuning
- Current: `batch_size=4`
- A100 can handle: `batch_size=16` of hoger
- More batches = better convergence

### Learning Rate Schedule
- Current: `learning_rate=0.001` (constant)
- Better: Add warmup (500 steps) + cosine annealing
- Zie `04_train_model_optimized.py` voor voorbeeld

### Monitoring
```python
# Install Weights & Biases (optional but recommended)
!pip install wandb
!wandb login

# In your training script:
import wandb
wandb.init(project="watergeusllm")
wandb.log({"loss": train_loss, "batch": batch_idx})
```

---

## Fase 5: Expected Performance

### GTX 1080 (Local)
- Speed: ~4.7 batches/sec
- Epoch 1: ~130 hours (5+ days)
- Total: ~38 days for 3 epochs

### A100 (Colab)
- Speed: ~50-100 batches/sec (10-20x faster)
- Epoch 1: ~6-8 hours
- Total: ~1-2 weeks for 3 epochs

**Cost:** ~$50-100 for Colab Pro subscription

---

## Troubleshooting

### Problem: Drive mount fails
```python
# Solution: Authenticate again
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### Problem: Out of memory on A100
```python
# Solution: Reduce batch size or use gradient accumulation
# In 04_train_model_optimized.py:
# config['accumulation_steps'] = 8  # Effective batch size = 4 * 8 = 32
```

### Problem: Git clone fails
```python
# Solution: Use HTTPS with token
# In Colab secrets, add GITHUB_TOKEN
!git clone https://{userdata.get('GITHUB_TOKEN')}@github.com/jdejong88-design/WGSNLD-V0.1.git
```

### Problem: Checkpoint incompatible
```python
# If checkpoint from different model config fails:
# Start fresh (no checkpoint), or retrain from checkpoint_batch_5000.pt
```

---

## Checklist für Colab Launch

- [ ] Drive folder structure created
- [ ] checkpoint_batch_50000.pt uploaded (~141 MB)
- [ ] tokens/ directory uploaded (~500 MB)
- [ ] dutch_tokenizer.json uploaded (<1 MB)
- [ ] GitHub repo verified (all scripts present)
- [ ] 02_tokenization.py completed locally
- [ ] Colab notebook cells prepared
- [ ] SSH or HTTPS auth working
- [ ] A100 GPU available in Colab
- [ ] Wandb account (optional) setup

---

**Ready to launch? Open Google Colab and paste Cell 1 above!**

🌊🚀
