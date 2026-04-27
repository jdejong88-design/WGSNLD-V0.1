#!/usr/bin/env python3
"""Train een Byte-Level BPE tokenizer op Nederlandse data."""

import logging
from pathlib import Path
from datasets import load_from_disk
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DutchBPETokenizer:
    """BPE Tokenizer trainer voor Nederlands."""

    def __init__(self, vocab_size=16000):
        """
        Initialize BPE tokenizer builder.

        Args:
            vocab_size: Grootte van vocabulaire (aanbevolen: 16k-32k voor Nederlands)
        """
        self.vocab_size = vocab_size
        logger.info(f"BPE Tokenizer initialized (vocab_size={vocab_size})")

    def create_tokenizer(self):
        """Maak een Byte-Level BPE tokenizer (GPT-2 stijl)."""
        tokenizer = Tokenizer(models.BPE())

        # Pre-tokenizer: ByteLevel splits op bytes, geen whitespace
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

        # Decoder: zet bytes terug naar tekst
        tokenizer.decoder = decoders.ByteLevel()

        # Post-processor: voeg speciale tekens toe
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

        return tokenizer

    def train_on_dataset(self, dataset_path, output_path='dutch_tokenizer.json'):
        """
        Train tokenizer op Nederlands dataset.

        Args:
            dataset_path: Path naar getokeniseerde dataset
            output_path: Waar tokenizer opslaan
        """
        # Load dataset
        logger.info(f"Loading dataset from {dataset_path}")
        dataset = load_from_disk(str(dataset_path))

        if hasattr(dataset, 'keys'):
            dataset = dataset['train']

        logger.info(f"Dataset size: {len(dataset)} samples")

        # Create tokenizer
        tokenizer = self.create_tokenizer()

        # Training data: extract text from dataset
        # We gebruiken de preprocessed teksten
        def batch_iterator(batch_size=1000):
            """Yield batches van tekst voor training."""
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
                if 'text' in batch:
                    yield batch['text']
                else:
                    # Als dataset alleen input_ids heeft, skip (dit is preprocessed)
                    logger.warning("Dataset contains no 'text' column. Using original Wikipedia data.")
                    break

        # Trainer: Byte-Level BPE (geen WordPiece ##-prefix)
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            show_progress=True,
            special_tokens=['<|endoftext|>'],
            min_frequency=2
        )

        # Train
        logger.info(f"Training BPE tokenizer with vocab_size={self.vocab_size}...")

        # We moeten de teksten naar files schrijven voor training
        # Dit is omdat tokenizers library files verwacht, niet dataset objects
        logger.info("Preparing training data...")

        wiki_path = Path('E:/Claude/workflow/WatergeusLLM/datasets/wikipedia_nl_sample_processed')

        if wiki_path.exists():
            # Gebruik preprocessed data
            wiki_data = load_from_disk(str(wiki_path))

            # Write to temporary file
            temp_file = Path('temp_training_data.txt')
            logger.info(f"Writing {len(wiki_data)} samples to {temp_file}")

            with open(temp_file, 'w', encoding='utf-8') as f:
                for sample in wiki_data['text']:
                    if sample:
                        f.write(sample + '\n')

            # Train op file
            logger.info("Training tokenizer...")
            tokenizer.train(
                files=[str(temp_file)],
                trainer=trainer
            )

            # Clean up
            temp_file.unlink()
            logger.info("Temporary file cleaned up")
        else:
            logger.error(f"Dataset not found at {wiki_path}")
            logger.info("Run 01_preprocessing.py first!")
            return False

        # Save tokenizer
        logger.info(f"Saving tokenizer to {output_path}")
        tokenizer.save(output_path)

        # Ook opslaan als vocab.json en merges.txt voor GPT-2 compatibiliteit
        output_dir = Path(output_path).parent
        vocab_path = output_dir / 'vocab.json'
        merges_path = output_dir / 'merges.txt'

        # Extract vocab van het getrainde model
        import json
        vocab = tokenizer.get_vocab()
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved vocab to {vocab_path} ({len(vocab)} tokens)")

        # Extract en opslaan merges.txt uit BPE model
        try:
            if hasattr(tokenizer.model, 'merges') and tokenizer.model.merges:
                with open(merges_path, 'w', encoding='utf-8') as f:
                    for merge_pair in tokenizer.model.merges:
                        f.write(f"{merge_pair}\n")
                logger.info(f"Saved merges to {merges_path} ({len(tokenizer.model.merges)} merges)")
            else:
                logger.warning("No merges found in BPE model. Merges file not created.")
        except Exception as e:
            logger.warning(f"Could not extract merges: {e}. Using tokenizer.json is sufficient.")

        logger.info("[OK] Tokenizer training complete!")
        return True

    def test_tokenizer(self, tokenizer_path='dutch_tokenizer.json', num_examples=5):
        """
        Test de getrainde tokenizer.

        Args:
            tokenizer_path: Path naar saved tokenizer
            num_examples: Aantal test voorbeelden
        """
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)

        # Test voorbeelden
        test_texts = [
            "Dit is een Nederlandse zin.",
            "Amsterdam is de hoofdstad van Nederland.",
            "Het Nederlands is een mooie taal.",
            "Machine learning is fascinerend!",
            "Artificial Intelligence verandert de wereld.",
        ]

        print("\n" + "="*70)
        print("TOKENIZER TEST RESULTATEN")
        print("="*70)

        print(f"\nVocabulaire grootte: {len(tokenizer.get_vocab()):,}")
        print(f"\nTest op {num_examples} Nederlandse zinnen:\n")

        for i, text in enumerate(test_texts[:num_examples], 1):
            encoded = tokenizer.encode(text)
            tokens = encoded.tokens
            token_ids = encoded.ids

            print(f"{i}. Input: {text}")
            print(f"   Tokens ({len(tokens)}): {tokens}")
            print(f"   Token IDs: {token_ids[:15]}{'...' if len(token_ids) > 15 else ''}")
            print()

        # Statistieken
        print("="*70)
        print("STATISTIEKEN")
        print("="*70)

        total_tokens = 0
        for text in test_texts:
            total_tokens += len(tokenizer.encode(text).ids)

        avg_tokens = total_tokens / len(test_texts)
        print(f"Gemiddeld aantal tokens per zin: {avg_tokens:.1f}")
        print(f"Vocabulaire grootte: {len(tokenizer.get_vocab()):,} tokens")


def main():
    """Main pipeline."""
    print("="*70)
    print("[*] BPE Tokenizer Training Pipeline - Nederlands")
    print("="*70)

    try:
        # Stap 1: Create trainer
        print("\n[1] INITIALIZING TOKENIZER\n")
        print("-"*50)

        trainer = DutchBPETokenizer(vocab_size=16000)
        print("[OK] BPE Tokenizer initialized")
        print("[OK] Vocabulaire grootte: 16.000 tokens")
        print("[OK] Model: Byte-Pair Encoding (BPE)")

        # Stap 2: Train tokenizer
        print("\n[2] TRAINING TOKENIZER\n")
        print("-"*50)

        dataset_path = Path('E:/Claude/workflow/WatergeusLLM/datasets/wikipedia_nl_sample_processed')
        output_path = 'E:/Claude/workflow/WatergeusLLM/dutch_tokenizer.json'

        success = trainer.train_on_dataset(
            dataset_path=dataset_path,
            output_path=output_path
        )

        if not success:
            logger.error("Training failed!")
            return

        # Stap 3: Test tokenizer
        print("\n[3] TESTING TOKENIZER\n")
        print("-"*50)

        trainer.test_tokenizer(
            tokenizer_path=output_path,
            num_examples=5
        )

        # Stap 4: Summary
        print("\n[4] SUMMARY\n")
        print("-"*50)

        output_file = Path(output_path)
        if output_file.exists():
            size_mb = output_file.stat().st_size / 1024 / 1024
            print(f"[OK] Tokenizer saved: {output_path}")
            print(f"[OK] File size: {size_mb:.1f} MB")
            print(f"[OK] Vocabulaire grootte: 16.000 tokens")
            print(f"\n[OK] BPE Tokenizer Training Complete!")
            print(f"Ready for Stap 3: Model Opzetten")
        else:
            logger.error("Tokenizer file not found!")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
