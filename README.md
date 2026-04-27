# WatergeusLLM

Een Nederlands talige Language Model gebaseerd op Transformer-architectuur.

## Project Overzicht

WatergeusLLM bouwt een **Nano-GPT** model op Nederlandse teksten (Wikipedia artikel-dataset). Het project volgt een complete pipeline van data-preparatie tot training.

### Stappen

1. **`01_preprocess.py`** — Wikipedia artikel-download en schoonmaak
2. **`02_tokenization.py`** — BPE tokenizer-training (16.000 vocab)
3. **`03_build_transformer_model.py`** — Nano-GPT architectuur (6 lagen, 256 hidden, 4 heads)
4. **`04_train_model.py`** — Model-training met checkpoint recovery
5. **`05_generate_text.py`** — Tekstgeneratie (nog te implementeren)

## Architectuur

- **Model:** Nano-GPT Transformer (causal language modeling)
- **Parameters:** ~30M
- **Tokenizer:** Custom BPE (16.000 tokens)
- **Training:** AdamW optimizer, cross-entropy loss, gradient clipping
- **Hardware:** GTX 1080 (8GB VRAM)

## Installatie

```bash
pip install -r requirements.txt
```

## Training

```bash
# Stage 1: Tokenizer bouwen
python 02_tokenization.py

# Stage 2: Model bouwen
python 03_build_transformer_model.py

# Stage 3: Training starten
python 04_train_model.py
```

Training draait met **checkpoint-recovery**: elke 5.000 batches wordt een checkpoint opgeslagen, en bij onderbreking (Ctrl+C) wordt `resume.pt` aangemaakt voor hervatting.

## Status

- ✅ Data preprocessing (50.000 samples)
- ✅ Tokenizer training (16.000 BPE vocab)
- ✅ Model architectuur (30M params, 6 layers)
- 🔄 **Training in progress** (Epoch 1/3, batch ~48.300/15.408.626)
- ✅ Checkpoint recovery system (every 5k batches)
- ⏳ Colab transition (A100 GPU training)
- ⏳ Tekstgeneratie (na training)

## Checkpoints & Data

Checkpoints worden opgeslagen in `checkpoints/` en zijn niet in GitHub opgenomen (te groot). Ze staan op:
- **Lokaal:** `E:/Claude/workflow/WatergeusLLM/checkpoints/`
- **Google Drive:** `MyDrive/WatergeusLLM/checkpoints/`

Tokenized data (`tokens/`) en preprocessed Arrow data ook niet in GitHub.

## Cloud Training (Google Colab)

Voor sneller training op A100 GPU:

```bash
# In Colab Cell 1:
!git clone https://github.com/jdejong88-design/WGSNLD-V0.1.git /content/repo
cd /content/repo

# In Colab Cell 2:
from google.colab import drive
drive.mount('/content/drive')

# In Colab Cell 3:
python colab_setup.py

# In Colab Cell 4:
python 04_train_model.py  # or 04_train_model_optimized.py for FP16
```

**Voordelen Colab:**
- A100 GPU: ~10x sneller dan GTX 1080
- Epoch 1: 5+ dagen (lokaal) → 6-8 uur (Colab)
- Mixed precision (FP16) beschikbaar
- Automatic checkpointing naar Drive

## GPU Optimization

Voor snellere training is `04_train_model_optimized.py` beschikbaar met:
- Mixed Precision (FP16)
- Gradient Accumulation
- ~50% geheugeninzet vs FP32

## Licentie

MIT
