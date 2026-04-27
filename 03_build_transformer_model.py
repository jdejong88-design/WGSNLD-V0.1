#!/usr/bin/env python3
"""Bouw een Nano-GPT Transformer-architectuur voor Nederlands."""

import logging
import torch
import torch.nn as nn
from pathlib import Path
from tokenizers import Tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    """Één Transformer encoder-blok met multi-head attention en feedforward."""

    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x, causal_mask=None):
        """Pas attention en feedforward toe met residual connections."""
        # Multi-head attention met residual
        attn_output, _ = self.attention(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + attn_output)

        # Feedforward met residual
        ff_output = self.feedforward(x)
        x = self.norm2(x + ff_output)

        return x


class NanoGPT(nn.Module):
    """Nano-GPT: Klein Transformer-model voor causal language modeling."""

    def __init__(self, vocab_size, hidden_size=512, num_layers=8, num_heads=8,
                 intermediate_size=2048, max_seq_length=512, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)

        # Transformer blokken
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])

        # Output layer with weight tying (share weights with token embeddings)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

        # Max sequence length (voor causal mask)
        self.max_seq_length = max_seq_length

        logger.info(f"NanoGPT initialized: {num_layers} lagen, {hidden_size} hidden size, {num_heads} heads")

    def _create_causal_mask(self, seq_length, device):
        """Creëer causal mask zodat tokens alleen naar vorige tokens kunnen kijken."""
        mask = torch.triu(torch.ones(seq_length, seq_length, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass voor causal language modeling.

        Args:
            input_ids: Token IDs (batch_size, seq_length)
            attention_mask: Optional mask voor padding

        Returns:
            logits: (batch_size, seq_length, vocab_size)
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        # Shape validatie – harde checks (niet assert, die worden genegeerd in -O mode)
        if seq_length > self.max_seq_length:
            raise ValueError(f"Sequence length ({seq_length}) overschrijdt het maximum ({self.max_seq_length}).")
        if input_ids.dtype != torch.long:
            raise ValueError(f"Input moet torch.long zijn, maar kreeg {input_ids.dtype}.")

        # Token embeddings
        token_embeds = self.token_embeddings(input_ids)

        # Position embeddings
        positions = torch.arange(seq_length, device=device).unsqueeze(0)
        pos_embeds = self.position_embeddings(positions)

        # Combineer en dropout
        x = self.embedding_dropout(token_embeds + pos_embeds)

        # Causal mask dinamisch genereren voor deze sequence length
        causal_mask = self._create_causal_mask(seq_length, device)

        # Pas Transformer blokken toe
        for block in self.transformer_blocks:
            x = block(x, causal_mask=causal_mask)

        # Output layer with weight tying
        x = self.layer_norm(x)
        logits = x @ self.token_embeddings.weight.T + self.output_bias

        return logits

    def count_parameters(self):
        """Tel het totale aantal traineerbare parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_tokenizer(tokenizer_path):
    """Laad de BPE tokenizer en geef vocab_size."""
    logger.info(f"Tokenizer laden uit {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = len(tokenizer.get_vocab())
    logger.info(f"Tokenizer geladen: {vocab_size} tokens in vocabulaire")
    return tokenizer, vocab_size


def main():
    """Bouw en initialiseer het Nano-GPT model."""
    print("="*70)
    print("[*] Nano-GPT Model Opzetten - Nederlands")
    print("="*70)

    try:
        # Stap 1: Tokenizer laden
        print("\n[1] TOKENIZER LADEN\n")
        print("-"*50)

        tokenizer_path = 'E:/Claude/workflow/WatergeusLLM/dutch_tokenizer.json'
        tokenizer, vocab_size = load_tokenizer(tokenizer_path)
        print(f"[OK] Tokenizer geladen")
        print(f"[OK] Vocabulaire grootte: {vocab_size} tokens")

        # Stap 2: Model configuratie
        print("\n[2] MODEL CONFIGURATIE\n")
        print("-"*50)

        # Geoptimaliseerd voor 8GB GPU: smaller model, stable training
        config = {
            'vocab_size': max(vocab_size, 16384),
            'hidden_size': 256,  # Gereduceerd van 512 (OOM op output layer)
            'num_layers': 6,     # Gereduceerd van 8
            'num_heads': 4,  # Gereduceerd van 8 (multi-head attention OOM op 8GB GPU)
            'intermediate_size': 2048,
            'max_seq_length': 512,
            'dropout': 0.1
        }

        print(f"[OK] Architectuur configuratie:")
        print(f"    • Lagen: {config['num_layers']}")
        print(f"    • Hidden size: {config['hidden_size']}")
        print(f"    • Attention heads: {config['num_heads']}")
        print(f"    • Intermediate size (feedforward): {config['intermediate_size']}")
        print(f"    • Max sequence length: {config['max_seq_length']}")

        # Stap 3: Model initialiseren
        print("\n[3] MODEL INITIALISEREN\n")
        print("-"*50)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = NanoGPT(**config)
        model.to(device)

        print(f"[OK] Nano-GPT model gecreëerd")
        print(f"[OK] Apparaat: {device.upper()}")

        # Stap 4: Parameter count
        print("\n[4] PARAMETERANALYSE\n")
        print("-"*50)

        total_params = model.count_parameters()
        total_params_millions = total_params / 1_000_000

        print(f"[OK] Totaal traineerbare parameters: {total_params:,}")
        print(f"[OK] In miljoen: {total_params_millions:.1f}M")

        # Verificatie
        if 30 < total_params_millions < 40:
            print(f"[OK] Parameter count is correct (verwacht: ~35M)")
        else:
            logger.warning(f"Parameter count is {total_params_millions:.1f}M, verwacht ~35M")

        # Breakdown per component (met weight tying)
        embedding_params = model.token_embeddings.weight.numel() + \
                          model.position_embeddings.weight.numel()
        transformer_params = sum(p.numel() for p in model.transformer_blocks.parameters())
        output_params = sum(p.numel() for p in model.layer_norm.parameters()) + \
                       model.output_bias.numel()

        print(f"\n[Breakdown per component]")
        print(f"  Embeddings: {embedding_params:,} ({embedding_params/total_params*100:.1f}%)")
        print(f"  Transformer blokken: {transformer_params:,} ({transformer_params/total_params*100:.1f}%)")
        print(f"  Output layer: {output_params:,} ({output_params/total_params*100:.1f}%)")

        # Stap 5: Model opslaan
        print("\n[5] MODEL OPSLAAN\n")
        print("-"*50)

        model_save_path = 'E:/Claude/workflow/WatergeusLLM/nano_gpt_model.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'vocab_size': vocab_size,
            'total_parameters': total_params
        }, model_save_path)

        logger.info(f"Model opgeslagen naar {model_save_path}")
        print(f"[OK] Model opgeslagen: {model_save_path}")

        # Stap 6: Samenvatting
        print("\n[6] SAMENVATTING\n")
        print("-"*50)

        print(f"[OK] Nano-GPT model succesvol gecreëerd!")
        print(f"[OK] Architectuur: {config['num_layers']} lagen × {config['hidden_size']} hidden × {config['num_heads']} heads")
        print(f"[OK] Parameters: {total_params_millions:.1f}M")
        print(f"[OK] Vocabulaire: {vocab_size} tokens (Nederlands BPE)")
        print(f"[OK] Ready voor Stap 4: Model Training")

    except FileNotFoundError as e:
        logger.error(f"Bestand niet gevonden: {e}")
        print(f"[FOUT] {e}")
        return False
    except Exception as e:
        logger.error(f"Fout: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
