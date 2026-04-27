#!/usr/bin/env python3
"""Hertokeniseer dataset met eigen BPE tokenizer (Dutch_tokenizer.json)."""

import logging
import torch
import json
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retokenize_dataset():
    """Hertokeniseer dataset met BPE tokenizer."""

    print("="*70)
    print("[*] Dataset Hertokeniseren met BPE Tokenizer")
    print("="*70)

    # Stap 1: Tokenizer laden
    print("\n[1] TOKENIZER LADEN\n")
    print("-"*50)

    tokenizer_path = 'dutch_tokenizer.json'
    if not Path(tokenizer_path).exists():
        print(f"[FOUT] Tokenizer niet gevonden: {tokenizer_path}")
        return False

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = len(tokenizer.get_vocab())
    print(f"[OK] Tokenizer geladen: {vocab_size} tokens")

    # Stap 2: Preprocessed dataset laden
    print("\n[2] PREPROCESSED DATASET LADEN\n")
    print("-"*50)

    processed_dir = Path('datasets/wikipedia_nl_sample_processed')
    if not processed_dir.exists():
        print(f"[FOUT] Dataset niet gevonden: {processed_dir}")
        return False

    # Laad Arrow dataset
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(str(processed_dir))
        print(f"[OK] Dataset geladen: {len(dataset)} artikelen")
    except Exception as e:
        print(f"[FOUT] Kan Arrow dataset niet laden: {e}")
        return False

    # Stap 3: Hertokeniseer en sla op
    print("\n[3] HERTOKENISEREN EN OPSLAAN\n")
    print("-"*50)

    output_dir = Path('datasets/wikipedia_nl_tokenized_bpe')
    output_dir.mkdir(parents=True, exist_ok=True)

    all_tokens = []
    max_token_id = 0
    batch_size = 100
    batch_idx = 0

    pbar = tqdm(dataset, desc="Tokeniseren")

    for article in pbar:
        try:
            text = article.get('text', '')

            if not text:
                continue

            # Tokeniseer
            encoding = tokenizer.encode(text)
            token_ids = encoding.ids

            if not token_ids:
                continue

            all_tokens.extend(token_ids)
            if token_ids:
                max_token_id = max(max_token_id, max(token_ids))

            # Batch opslaan elke 100 artikelen
            if len(all_tokens) >= batch_size * 512:  # 512 is seq_length
                tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)
                output_file = output_dir / f'tokens_{batch_idx:05d}.pt'
                torch.save(tokens_tensor, output_file)
                batch_idx += 1
                all_tokens = []

        except Exception as e:
            logger.warning(f"Fout bij artikel: {e}")
            continue

    # Sla resterende tokens op
    if all_tokens:
        tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)
        output_file = output_dir / f'tokens_{batch_idx:05d}.pt'
        torch.save(tokens_tensor, output_file)
        batch_idx += 1

    print(f"\n[OK] {batch_idx} batch bestanden opgeslagen")

    # Stap 4: Verificatie
    print("\n[4] VERIFICATIE\n")
    print("-"*50)

    print(f"[OK] Max token ID: {max_token_id}")
    if max_token_id < vocab_size:
        print(f"[OK] Alle tokens in bereik (0..{vocab_size-1})")
    else:
        print(f"[FOUT] Token ID {max_token_id} valt buiten bereik!")
        return False

    pt_files = sorted(output_dir.glob('tokens_*.pt'))
    print(f"[OK] Total batch bestanden: {len(pt_files)}")

    # Controleer totaal tokens
    total_tokens = 0
    for pt_file in pt_files:
        tokens = torch.load(pt_file)
        total_tokens += len(tokens)

    print(f"[OK] Totaal tokens: {total_tokens:,}")

    # Sla metadata op
    metadata = {
        'tokenizer': 'dutch_tokenizer.json',
        'vocab_size': vocab_size,
        'max_token_id': max_token_id,
        'total_tokens': total_tokens,
        'num_batches': len(pt_files),
        'source': 'wikipedia_nl_sample_processed'
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Metadata opgeslagen")

    # Samenvatting
    print("\n[5] SAMENVATTING\n")
    print("-"*50)
    print(f"✓ Dataset hertokeniseerd met BPE tokenizer")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Batch bestanden: {len(pt_files)}")
    print(f"✓ Totaal tokens: {total_tokens:,}")
    print(f"✓ Gereed voor training met 04_train_model.py")

    return True


if __name__ == '__main__':
    success = retokenize_dataset()
    if not success:
        print("\n[FOUT] Hertokenisering mislukt")
        exit(1)
    else:
        print("\n[OK] Hertokenisering voltooid!")
