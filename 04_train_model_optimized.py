#!/usr/bin/env python3
"""
OPTIMIZED Training: Mixed Precision (FP16) + Gradient Accumulation
Versie voor Na Epoch 1 — extra snelheid + geheugenefficiency
"""

import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm
import json
from datetime import datetime
import sys
import os

# CUDA memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# OPTIMIZATIE: Mixed Precision setup
# ============================================================================
from torch.amp import autocast, GradScaler

scaler = GradScaler()  # Voor numerieke stabiliteit bij FP16


class TokenizedDataset(Dataset):
    """Laad tokenized sequenties uit .pt bestanden."""

    def __init__(self, tokenized_dir, seq_length=512):
        self.seq_length = seq_length
        self.dataset = None
        self.tokens = None

        tokenized_path = Path(tokenized_dir)
        token_files = sorted(tokenized_path.glob("tokens_*.pt"))

        if token_files:
            logger.info(f"Token bestanden gevonden: {len(token_files)}")
            all_tokens = []
            for token_file in token_files:
                batch_tokens = torch.load(token_file)
                if isinstance(batch_tokens, torch.Tensor):
                    all_tokens.extend(batch_tokens.tolist())
                else:
                    all_tokens.extend(batch_tokens)
            self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        else:
            logger.warning(f"Geen token bestanden gevonden in {tokenized_dir}")
            self.tokens = torch.tensor([], dtype=torch.long)

        if self.tokens is not None:
            logger.info(f"Totaal tokens geladen: {len(self.tokens):,}")

    def __len__(self):
        if self.tokens is not None:
            return max(0, len(self.tokens) - self.seq_length)
        return 0

    def __getitem__(self, idx):
        if self.tokens is not None:
            input_ids = self.tokens[idx:idx + self.seq_length]
            target_ids = self.tokens[idx + 1:idx + self.seq_length + 1]
        else:
            return torch.zeros(self.seq_length, dtype=torch.long), torch.zeros(self.seq_length, dtype=torch.long)

        if len(input_ids) < self.seq_length:
            padding = torch.zeros(self.seq_length - len(input_ids), dtype=torch.long)
            input_ids = torch.cat([input_ids, padding])
            target_ids = torch.cat([target_ids, padding])

        return input_ids, target_ids


def train_epoch_optimized(model, train_loader, optimizer, device, epoch, num_epochs, accumulation_steps=4):
    """
    Train één epoch met:
    - Mixed Precision (FP16)
    - Gradient Accumulation
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [FP16+AccGrad]", leave=True)

    optimizer.zero_grad()  # Start met zero gradients

    for batch_idx, (input_ids, target_ids) in enumerate(pbar):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # ====================================================================
        # OPTIMIZATIE: Autocast context — automatisch FP16 waar mogelijk
        # ====================================================================
        with autocast(device_type='cuda'):
            logits = model(input_ids)

            # Loss berekenen
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=0
            )

            # Gradient Accumulation: schaal loss naar durchsnit
            loss = loss / accumulation_steps

        # ====================================================================
        # OPTIMIZATIE: GradScaler voor numerieke stabiliteit met FP16
        # ====================================================================
        scaler.scale(loss).backward()

        # Accumulation stap
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Optimizer stap
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps  # Unscale voor tracking
        num_batches += 1

        avg_loss = total_loss / num_batches
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "dtype": "FP16"})

        if (batch_idx + 1) % 100 == 0:
            logger.info(f"Batch {batch_idx+1}: Loss = {avg_loss:.4f} [FP16 optimized]")

    return total_loss / num_batches


def validate(model, val_loader, device):
    """Validatie met Mixed Precision."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for input_ids, target_ids in tqdm(val_loader, desc="Validatie [FP16]"):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            with autocast(device_type='cuda'):
                logits = model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                    ignore_index=0
                )

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def load_model_and_config(model_path):
    """Laad model state dict en config."""
    checkpoint = torch.load(model_path, map_location='cpu')
    return checkpoint['model_state_dict'], checkpoint['config']


def main():
    print("="*70)
    print("[*] Nano-GPT Model Training - OPTIMIZED (FP16 + Grad Accumulation)")
    print("="*70)

    try:
        print("\n[1] TRAINING CONFIGURATIE (OPTIMIZED)\n")
        print("-"*50)

        config = {
            'learning_rate': 0.001,
            'batch_size': 4,  # Origineel
            'accumulation_steps': 4,  # NIEUW: virtueel batch_size = 4 × 4 = 16
            'num_epochs': 3,
            'num_warmup_steps': 500,
            'gradient_clip': 1.0,
            'val_split': 0.1,
            'seq_length': 512,
            'use_fp16': True,  # NIEUW
        }

        print(f"[OK] Batch size: {config['batch_size']} (FP32)")
        print(f"[OK] Accumulation steps: {config['accumulation_steps']}")
        print(f"[OK] Virtuele batch size: {config['batch_size'] * config['accumulation_steps']}")
        print(f"[OK] Mixed Precision: ENABLED (FP16)")
        print(f"[OK] Epochs: {config['num_epochs']}")

        print("\n[2] MODEL LADEN\n")
        print("-"*50)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[OK] Apparaat: {device.upper()}")
        print(f"[OK] CUDA Compute Capability: {torch.cuda.get_device_capability(0) if device == 'cuda' else 'N/A'}")

        model_path = 'E:/Claude/workflow/WatergeusLLM/nano_gpt_model.pt'
        if not Path(model_path).exists():
            print(f"[FOUT] Model niet gevonden: {model_path}")
            return False

        state_dict, model_config = load_model_and_config(model_path)

        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("model", "E:/Claude/workflow/WatergeusLLM/03_build_transformer_model.py")
        model_module = module_from_spec(spec)
        spec.loader.exec_module(model_module)

        model = model_module.NanoGPT(**model_config)
        model.load_state_dict(state_dict)
        model.to(device)

        # OPTIMIZATIE: Model in FP16 casting waar mogelijk
        if config['use_fp16']:
            model = model.half()  # Cast naar float16
            print(f"[OK] Model cast naar FP16")

        total_params = model.count_parameters()
        print(f"[OK] Parameters: {total_params/1e6:.1f}M")

        print("\n[3] DATASET LADEN\n")
        print("-"*50)

        tokenized_dir = 'E:/Claude/workflow/WatergeusLLM/datasets/wikipedia_nl_tokenized_bpe'

        if not Path(tokenized_dir).exists():
            print(f"[FOUT] Tokenized dataset niet gevonden: {tokenized_dir}")
            return False

        dataset = TokenizedDataset(tokenized_dir, seq_length=config['seq_length'])
        print(f"[OK] Dataset grootte: {len(dataset):,} sequenties")

        val_size = int(len(dataset) * config['val_split'])
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        print(f"[OK] Training samples: {len(train_dataset):,}")
        print(f"[OK] Validatie samples: {len(val_dataset):,}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0
        )

        print(f"[OK] Training batches: {len(train_loader)}")
        print(f"[OK] Validatie batches: {len(val_loader)}")

        print("\n[4] OPTIMIZER INSTELLEN\n")
        print("-"*50)

        optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=0.01
        )

        print(f"[OK] Optimizer: AdamW (lr={config['learning_rate']})")
        print(f"[OK] GradScaler active voor FP16 numerieke stabiliteit")

        print("\n[5] MODEL TRAINING (OPTIMIZED)\n")
        print("-"*50)

        training_history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'optimization': 'FP16 + Gradient Accumulation',
            'timestamp': datetime.now().isoformat()
        }

        best_val_loss = float('inf')
        patience = 2
        patience_counter = 0

        for epoch in range(config['num_epochs']):
            train_loss = train_epoch_optimized(
                model, train_loader, optimizer, device, epoch, config['num_epochs'],
                accumulation_steps=config['accumulation_steps']
            )

            val_loss = validate(model, val_loader, device)

            training_history['epochs'].append(epoch + 1)
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)

            print(f"\n[EPOCH {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                checkpoint_path = 'E:/Claude/workflow/WatergeusLLM/checkpoints/best_model_optimized.pt'
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'config': model_config,
                    'optimization': 'FP16+AccGrad'
                }, checkpoint_path)

                print(f"[OK] Best model checkpoint opgeslagen (optimized)")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[INFO] Early stopping na {patience} epochs zonder verbetering")
                    break

        print("\n[6] RESULTATEN OPSLAAN\n")
        print("-"*50)

        history_path = 'E:/Claude/workflow/WatergeusLLM/training_history_optimized.json'
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)

        print(f"[OK] Training history opgeslagen: {history_path}")

        print("\n[7] TRAINING VOLTOOID\n")
        print("-"*50)
        print(f"[OK] Beste validatie loss: {best_val_loss:.4f}")
        print(f"[OK] Optimization: FP16 + Gradient Accumulation")
        print(f"[OK] Memory efficiency: ~50% besparing vs FP32")
        print(f"[OK] Speed improvement: Expected 20-40% sneller")

        return True

    except Exception as e:
        logger.error(f"Fout: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
