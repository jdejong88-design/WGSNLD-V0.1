#!/usr/bin/env python3
"""Tokenization voor Nederlandse LLM datasets."""

import logging
from pathlib import Path
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Nederlandse tokenizer opties
TOKENIZERS = {
    'robbert': 'pdelobelle/robbert-v2-dutch-base',
    'gpt2': 'gpt2',
    'bert-base': 'bert-base-multilingual-cased',
}

class DutchTokenizer:
    """Dutch text tokenization."""

    def __init__(self, model_name='robbert'):
        """
        Initialize tokenizer.

        Args:
            model_name: Key from TOKENIZERS dict
        """
        if model_name not in TOKENIZERS:
            raise ValueError(f"Unknown tokenizer: {model_name}")

        model_id = TOKENIZERS[model_name]
        logger.info(f"Loading {model_name} tokenizer: {model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model_name = model_name
        self.model_id = model_id

    def tokenize_text(self, text, max_length=512, truncation=True):
        """
        Tokenize single text.

        Args:
            text: Input text
            max_length: Maximum token length
            truncation: Whether to truncate

        Returns:
            Dict with input_ids, attention_mask, token_type_ids
        """
        return self.tokenizer(
            text,
            max_length=max_length,
            truncation=truncation,
            padding=False,
            return_tensors=None,
        )

    def tokenize_dataset(self, dataset, text_column='text', max_length=512, batch_size=100):
        """
        Tokenize entire dataset.

        Args:
            dataset: Hugging Face dataset
            text_column: Name of text column
            max_length: Max token length
            batch_size: Batch processing size

        Returns:
            Tokenized dataset
        """
        def tokenize_function(examples):
            """Tokenize batch."""
            return self.tokenizer(
                examples[text_column],
                max_length=max_length,
                truncation=True,
                padding='max_length',
            )

        logger.info(f"Tokenizing {len(dataset)} samples...")

        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=batch_size,
            remove_columns=[text_column],
            desc="Tokenizing",
        )

        return tokenized

    def get_vocab_size(self):
        """Get vocabulary size."""
        return len(self.tokenizer)

    def get_token_stats(self, texts, sample_size=None):
        """
        Get tokenization statistics.

        Args:
            texts: List of texts
            sample_size: Number of texts to sample

        Returns:
            Dict with statistics
        """
        if sample_size:
            texts = texts[:sample_size]

        token_lengths = []

        for text in tqdm(texts, desc="Computing stats"):
            tokens = self.tokenizer.encode(text)
            token_lengths.append(len(tokens))

        avg_length = sum(token_lengths) / len(token_lengths)
        min_length = min(token_lengths)
        max_length = max(token_lengths)

        return {
            'total_texts': len(texts),
            'avg_token_length': avg_length,
            'min_token_length': min_length,
            'max_token_length': max_length,
            'total_tokens': sum(token_lengths),
        }


def main():
    """Main tokenization pipeline."""
    print("=" * 70)
    print("[*] Dutch Text Tokenization Pipeline")
    print("=" * 70)

    # 1. Tokenizer comparison
    print("\n[1] TOKENIZER COMPARISON\n")

    example_texts = [
        "Dit is een normale Nederlandse zin.",
        "Amsterdam is de hoofdstad van Nederland.",
        "Het Nederlands is een West-Germaanse taal.",
    ]

    for tokenizer_name in ['robbert', 'gpt2']:
        try:
            print(f"\n{tokenizer_name.upper()}:")
            print("-" * 50)

            tok = DutchTokenizer(tokenizer_name)
            print(f"Model: {tok.model_id}")
            print(f"Vocab size: {tok.get_vocab_size():,}")

            for i, text in enumerate(example_texts[:2], 1):
                result = tok.tokenize_text(text)
                tokens = tok.tokenizer.convert_ids_to_tokens(result['input_ids'])

                print(f"\n  {i}. Text: {text}")
                print(f"     Tokens: {len(result['input_ids'])}")
                print(f"     IDs: {result['input_ids'][:10]}...")
                print(f"     Tokens: {tokens[:10]}...")

        except Exception as e:
            print(f"Error loading {tokenizer_name}: {e}")

    # 2. Process preprocessed dataset
    print("\n" + "=" * 70)
    print("[2] TOKENIZING WIKIPEDIA SAMPLE\n")

    try:
        # Load preprocessed data
        wiki_path = Path('E:/Claude/workflow/WatergeusLLM/datasets/wikipedia_nl_sample_processed')

        if not wiki_path.exists():
            logger.warning(f"Preprocessed dataset not found at {wiki_path}")
            logger.info("Run 01_preprocessing.py first!")
            return

        wiki_dataset = load_from_disk(str(wiki_path))
        logger.info(f"Loaded {len(wiki_dataset)} samples")

        # Initialize tokenizer
        tok = DutchTokenizer('robbert')

        # Show statistics
        print("\n[STATISTICS]")
        print("-" * 50)

        stats = tok.get_token_stats(wiki_dataset['text'], sample_size=100)

        print(f"Total texts: {stats['total_texts']}")
        print(f"Avg token length: {stats['avg_token_length']:.1f}")
        print(f"Min tokens: {stats['min_token_length']}")
        print(f"Max tokens: {stats['max_token_length']}")
        print(f"Total tokens: {stats['total_tokens']:,}")

        # Tokenize full dataset
        print("\n[TOKENIZING FULL DATASET]")
        print("-" * 50)

        tokenized_dataset = tok.tokenize_dataset(
            wiki_dataset,
            text_column='text',
            max_length=512,
            batch_size=100
        )

        print(f"\n[OK] Tokenization complete!")
        print(f"  Samples: {len(tokenized_dataset)}")
        print(f"  Columns: {tokenized_dataset.column_names}")

        # Show example
        print(f"\n[EXAMPLE] Tokenized sample:")
        example = tokenized_dataset[0]
        print(f"  Input IDs: {example['input_ids'][:20]}...")
        print(f"  Attention mask: {example['attention_mask'][:20]}...")

        if 'token_type_ids' in example:
            print(f"  Token type IDs: {example['token_type_ids'][:20]}...")

        # Save tokenized dataset
        output_path = Path('E:/Claude/workflow/WatergeusLLM/datasets/wikipedia_nl_tokenized')
        tokenized_dataset.save_to_disk(str(output_path))
        print(f"\n[OK] Tokenized dataset saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
