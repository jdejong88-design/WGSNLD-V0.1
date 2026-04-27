#!/usr/bin/env python3
"""Traineer het Nano-GPT model op 50k Nederlandse artikelen."""

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
import signal

# CUDA memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Graceful shutdown variables
_shutdown_signal = False
_current_state = {}

def signal_handler(signum, frame):
    """Save resume checkpoint on Ctrl+C"""
    global _shutdown_signal, _current_state
    _shutdown_signal = True
    if _current_state:
        resume_path = 'E:/Claude/workflow/WatergeusLLM/checkpoints/resume.pt'
        Path(resume_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(_current_state, resume_path)
        logger.info(f"[SHUTDOWN] Resume checkpoint opgeslagen op epoch {_current_state.get('epoch')}, batch {_current_state.get('batch')}")
    print("\n[INFO] Training onderbroken. Resume checkpoint beschikbaar.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


class TokenizedDataset(Dataset):
    """Laad tokenized sequenties uit Arrow of .pt bestanden."""

    def __init__(self, tokenized_dir, seq_length=512):
        self.seq_length = seq_length
        self.dataset = None
        self.tokens = None

        tokenized_path = Path(tokenized_dir)
        token_files = sorted(tokenized_path.glob("tokens_*.pt"))

        # Poging 1: .pt bestanden
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

        # Poging 2: HuggingFace Arrow dataset
        elif (tokenized_path / "data-00000-of-00001.arrow").exists():
            logger.info("Arrow file gevonden, laden via HuggingFace...")
            try:
                from datasets import load_from_disk
                hf_dataset = load_from_disk(str(tokenized_path))
                # HuggingFace dataset wrapper
                self.dataset = hf_dataset
                logger.info(f"HuggingFace dataset geladen")
            except Exception as e:
                logger.error(f"Kan HuggingFace dataset niet laden: {e}")
                self.tokens = torch.tensor([], dtype=torch.long)
        else:
            logger.warning(f"Geen token bestanden gevonden in {tokenized_dir}")
            self.tokens = torch.tensor([], dtype=torch.long)

        if self.tokens is not None:
            logger.info(f"Totaal tokens geladen: {len(self.tokens):,}")
        elif self.dataset is not None:
            logger.info(f"HuggingFace dataset items: {len(self.dataset):,}")

    def __len__(self):
        if self.tokens is not None:
            return max(0, len(self.tokens) - self.seq_length)
        elif self.dataset is not None:
            total_ids = sum(len(item['input_ids']) for item in self.dataset)
            return max(0, total_ids - self.seq_length)
        return 0

    def __getitem__(self, idx):
        if self.tokens is not None:
            input_ids = self.tokens[idx:idx + self.seq_length]
            target_ids = self.tokens[idx + 1:idx + self.seq_length + 1]
        elif self.dataset is not None:
            # Flatten all input_ids from dataset
            all_ids = []
            for item in self.dataset:
                all_ids.extend(item['input_ids'])
            all_ids_tensor = torch.tensor(all_ids, dtype=torch.long)
            input_ids = all_ids_tensor[idx:idx + self.seq_length]
            target_ids = all_ids_tensor[idx + 1:idx + self.seq_length + 1]
        else:
            return torch.zeros(self.seq_length, dtype=torch.long), torch.zeros(self.seq_length, dtype=torch.long)

        # Pad if necessary
        if len(input_ids) < self.seq_length:
            padding = torch.zeros(self.seq_length - len(input_ids), dtype=torch.long)
            input_ids = torch.cat([input_ids, padding])
            target_ids = torch.cat([target_ids, padding])

        return input_ids, target_ids


def train_epoch(model, train_loader, optimizer, device, epoch, num_epochs, model_config):
    """Train één epoch met tussentijdse checkpoints om de 5000 batches."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

    for batch_idx, (input_ids, target_ids) in enumerate(pbar):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # Forward pass
        logits = model(input_ids)

        # Loss berekenen
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=0
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        avg_loss = total_loss / num_batches
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # Update global state voor graceful shutdown (resume.pt)
        global _current_state
        _current_state = {
            'epoch': epoch,
            'batch': batch_idx + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': model_config
        }

        # Log elke 100 batches (met gradient norm monitoring)
        if (batch_idx + 1) % 100 == 0:
            logger.info(f"Batch {batch_idx+1}: Loss = {avg_loss:.4f}, Grad Norm = {grad_norm:.4f}")

        # NIEUW: Checkpoint om de 5000 batches (verzekering tegen crashes)
        if (batch_idx + 1) % 5000 == 0:
            checkpoint_path = f'E:/Claude/workflow/WatergeusLLM/checkpoints/checkpoint_batch_{batch_idx+1}.pt'
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'batch': batch_idx + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_loss': avg_loss,
                'config': model_config
            }, checkpoint_path)
            logger.info(f"[CHECKPOINT] Batch {batch_idx+1} opgeslagen naar {checkpoint_path}")

    return total_loss / num_batches


def validate(model, val_loader, device):
    """Validatie op held-out data."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for input_ids, target_ids in tqdm(val_loader, desc="Validatie"):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

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
    print("[*] Nano-GPT Model Training - Nederlands")
    print("="*70)

    try:
        # Stap 1: Configuratie
        print("\n[1] TRAINING CONFIGURATIE\n")
        print("-"*50)

        config = {
            'learning_rate': 0.001,
            'batch_size': 4,  # Terug naar 4 (batch_size=8 geeft OOM)
            'num_epochs': 3,
            'num_warmup_steps': 500,
            'gradient_clip': 1.0,
            'val_split': 0.1,
            'seq_length': 512,
        }

        print(f"[OK] Learning rate: {config['learning_rate']}")
        print(f"[OK] Batch size: {config['batch_size']}")
        print(f"[OK] Epochs: {config['num_epochs']}")
        print(f"[OK] Sequence length: {config['seq_length']}")

        # Stap 2: Model en device
        print("\n[2] MODEL LADEN\n")
        print("-"*50)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[OK] Apparaat: {device.upper()}")

        model_path = 'E:/Claude/workflow/WatergeusLLM/nano_gpt_model.pt'
        if not Path(model_path).exists():
            print(f"[FOUT] Model niet gevonden: {model_path}")
            return False

        state_dict, model_config = load_model_and_config(model_path)
        print(f"[OK] Model config geladen: {model_config['hidden_size']}D × {model_config['num_layers']} lagen")

        # Import model class
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("model", "E:/Claude/workflow/WatergeusLLM/03_build_transformer_model.py")
        model_module = module_from_spec(spec)
        spec.loader.exec_module(model_module)

        model = model_module.NanoGPT(**model_config)
        model.load_state_dict(state_dict)
        model.to(device)

        total_params = model.count_parameters()
        print(f"[OK] Parameters: {total_params/1e6:.1f}M")

        # Stap 3: Data laden
        print("\n[3] DATASET LADEN\n")
        print("-"*50)

        tokenized_dir = 'E:/Claude/workflow/WatergeusLLM/datasets/wikipedia_nl_tokenized_bpe'

        if not Path(tokenized_dir).exists():
            print(f"[FOUT] Tokenized dataset niet gevonden: {tokenized_dir}")
            print("[INFO] Voer eerst uit: python 02_tokenization.py")
            return False

        dataset = TokenizedDataset(tokenized_dir, seq_length=config['seq_length'])
        print(f"[OK] Dataset grootte: {len(dataset):,} sequenties")

        # Train/val split
        val_size = int(len(dataset) * config['val_split'])
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        print(f"[OK] Training samples: {len(train_dataset):,}")
        print(f"[OK] Validatie samples: {len(val_dataset):,}")

        # DataLoaders
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

        # Stap 4: Optimizer
        print("\n[4] OPTIMIZER INSTELLEN\n")
        print("-"*50)

        optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=0.01
        )

        print(f"[OK] Optimizer: AdamW (lr={config['learning_rate']})")

        # Stap 5: Training loop
        print("\n[5] MODEL TRAINING\n")
        print("-"*50)

        training_history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'timestamp': datetime.now().isoformat()
        }

        best_val_loss = float('inf')
        patience = 2
        patience_counter = 0

        # Check voor resume checkpoint
        resume_path = 'E:/Claude/workflow/WatergeusLLM/checkpoints/resume.pt'
        start_epoch = 0
        if Path(resume_path).exists():
            print("\n[*] Resume checkpoint gevonden, laden...")
            resume_data = torch.load(resume_path, map_location=device)
            model.load_state_dict(resume_data['model_state_dict'])
            optimizer.load_state_dict(resume_data['optimizer_state_dict'])
            start_epoch = resume_data.get('epoch', 0)
            logger.info(f"[RESUME] Hervat van epoch {start_epoch}")

        for epoch in range(start_epoch, config['num_epochs']):
            # Training
            train_loss = train_epoch(
                model, train_loader, optimizer, device, epoch, config['num_epochs'], model_config
            )

            # Validatie
            val_loss = validate(model, val_loader, device)

            # Logging
            training_history['epochs'].append(epoch + 1)
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)

            print(f"\n[EPOCH {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Checkpoint besten model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                checkpoint_path = 'E:/Claude/workflow/WatergeusLLM/checkpoints/best_model.pt'
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'config': model_config
                }, checkpoint_path)

                print(f"[OK] Best model checkpoint opgeslagen")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[INFO] Early stopping na {patience} epochs zonder verbetering")
                    break

        # Stap 6: Training history opslaan
        print("\n[6] RESULTATEN OPSLAAN\n")
        print("-"*50)

        history_path = 'E:/Claude/workflow/WatergeusLLM/training_history.json'
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)

        print(f"[OK] Training history opgeslagen: {history_path}")

        # Stap 7: Samenvatting
        print("\n[7] TRAINING VOLTOOID\n")
        print("-"*50)

        print(f"[OK] Beste validatie loss: {best_val_loss:.4f}")
        print(f"[OK] Model checkpoint: E:/Claude/workflow/WatergeusLLM/checkpoints/best_model.pt")
        print(f"[OK] Training history: {history_path}")
        print(f"[OK] Klaar voor Stap 5: Tekstgeneratie")

        return True

    except FileNotFoundError as e:
        logger.error(f"Bestand niet gevonden: {e}")
        print(f"[FOUT] {e}")
        return False
    except Exception as e:
        logger.error(f"Fout: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
