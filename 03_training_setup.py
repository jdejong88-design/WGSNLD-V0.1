#!/usr/bin/env python3
"""Training pipeline voor Nederlandse LLM."""

import logging
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model opties
MODELS = {
    'gpt2': 'gpt2',
    'gpt2-medium': 'gpt2-medium',
    'robbert': 'pdelobelle/robbert-v2-dutch-base',
    'distilbert': 'distilbert-base-multilingual-cased',
}


class DutchLLMTrainer:
    """Training pipeline voor Nederlands LLM."""

    def __init__(self, model_name='gpt2', device=None):
        """
        Initialize trainer.

        Args:
            model_name: Model key from MODELS dict
            device: torch device (auto-detect if None)
        """
        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        self.model_name = model_name
        self.model_id = MODELS[model_name]
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Loading model: {self.model_id}")
        logger.info(f"Device: {self.device}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Model loaded: {self.model_id}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def setup_training(self, dataset, output_dir='./checkpoints',
                      learning_rate=5e-5, batch_size=8, num_epochs=3,
                      warmup_steps=500, weight_decay=0.01):
        """
        Setup training configuration.

        Args:
            dataset: Tokenized dataset
            output_dir: Where to save checkpoints
            learning_rate: Learning rate
            batch_size: Batch size
            num_epochs: Number of training epochs
            warmup_steps: Warmup steps
            weight_decay: Weight decay

        Returns:
            Trainer instance
        """
        # Split dataset
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Eval samples: {len(eval_dataset)}")

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # No masked language modeling for causal LM
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            save_steps=100,
            eval_steps=100,
            logging_steps=50,
            evaluation_strategy='steps',
            save_strategy='steps',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            push_to_hub=False,
            seed=42,
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        return trainer, training_args

    def train(self, trainer):
        """Train the model."""
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training complete!")

    def save_model(self, output_dir):
        """Save trained model and tokenizer."""
        logger.info(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("Model saved!")


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("[*] Dutch LLM Training Pipeline")
    print("=" * 70)

    try:
        # Load tokenized dataset
        dataset_path = Path('E:/Claude/workflow/WatergeusLLM/datasets/wikipedia_nl_tokenized')

        if not dataset_path.exists():
            logger.error(f"Dataset not found at {dataset_path}")
            logger.info("Run 02_tokenization.py first!")
            return

        logger.info(f"Loading dataset from {dataset_path}")
        dataset = load_from_disk(str(dataset_path))
        logger.info(f"Loaded {len(dataset)} samples")

        # Initialize trainer
        print("\n[1] MODEL SETUP\n")
        print("-" * 50)

        trainer_obj = DutchLLMTrainer(model_name='gpt2')
        print(f"Model: {trainer_obj.model_id}")
        print(f"Device: {trainer_obj.device}")
        print(f"Vocab size: {len(trainer_obj.tokenizer):,}")

        # Setup training
        print("\n[2] TRAINING CONFIGURATION\n")
        print("-" * 50)

        output_dir = Path('E:/Claude/workflow/WatergeusLLM/checkpoints')
        output_dir.mkdir(parents=True, exist_ok=True)

        trainer, training_args = trainer_obj.setup_training(
            dataset=dataset,
            output_dir=str(output_dir),
            learning_rate=5e-5,
            batch_size=8,
            num_epochs=3,
            warmup_steps=100,
            weight_decay=0.01,
        )

        print(f"Learning rate: {training_args.learning_rate}")
        print(f"Batch size: {training_args.per_device_train_batch_size}")
        print(f"Epochs: {training_args.num_train_epochs}")
        print(f"Warmup steps: {training_args.warmup_steps}")
        print(f"Weight decay: {training_args.weight_decay}")
        print(f"Output dir: {output_dir}")

        # Show training summary
        print("\n[3] TRAINING SUMMARY\n")
        print("-" * 50)

        total_steps = len(trainer.get_train_dataloader()) * training_args.num_train_epochs
        print(f"Total training steps: {int(total_steps)}")
        print(f"Save every: {training_args.save_steps} steps")
        print(f"Eval every: {training_args.eval_steps} steps")

        # Train model
        print("\n[4] STARTING TRAINING\n")
        print("-" * 50)

        trainer_obj.train(trainer)

        # Save final model
        print("\n[5] SAVING MODEL\n")
        print("-" * 50)

        final_model_dir = Path('E:/Claude/workflow/WatergeusLLM/models/dutch_llm_gpt2')
        trainer_obj.save_model(str(final_model_dir))

        print(f"\n[OK] Training pipeline complete!")
        print(f"  Checkpoints: {output_dir}")
        print(f"  Final model: {final_model_dir}")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
