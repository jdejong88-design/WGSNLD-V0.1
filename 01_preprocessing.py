#!/usr/bin/env python3
"""Data Preprocessing voor Nederlandse LLM datasets."""

import re
import unicodedata
from pathlib import Path
from datasets import load_from_disk, Dataset
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DutchTextPreprocessor:
    """Nederlandse tekst preprocessing."""

    def __init__(self):
        # Regex patterns
        self.url_pattern = re.compile(r'http\S+|www\S+|ftp\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.html_pattern = re.compile(r'<[^>]+>')
        self.extra_spaces = re.compile(r'\s+')
        self.multiple_newlines = re.compile(r'\n{3,}')

    def remove_html(self, text):
        """Verwijder HTML/XML tags."""
        return self.html_pattern.sub('', text)

    def remove_urls(self, text):
        """Verwijder URLs."""
        return self.url_pattern.sub('', text)

    def remove_emails(self, text):
        """Verwijder email adressen."""
        return self.email_pattern.sub('', text)

    def normalize_whitespace(self, text):
        """Normaliseer whitespace."""
        # Newlines vervangen door spatie
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        # Multiple spaces naar single space
        text = self.extra_spaces.sub(' ', text)
        return text.strip()

    def normalize_unicode(self, text):
        """Normaliseer Unicode karakters."""
        # NFC normalisatie
        text = unicodedata.normalize('NFC', text)
        return text

    def remove_special_chars(self, text, keep_punctuation=True):
        """Verwijder ongewenste special characters."""
        if keep_punctuation:
            # Behoud: letters, cijfers, spaties, en basale punctuatie
            text = re.sub(r'[^\w\s.!?,;:\'\"\-–—()—/&%€$]', '', text)
        else:
            # Verwijder alles behalve letters, cijfers, spaties
            text = re.sub(r'[^\w\s]', '', text)
        return text

    def remove_control_chars(self, text):
        """Verwijder control characters."""
        text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
        return text

    def is_valid_text(self, text, min_length=10, min_words=3):
        """Controleer of tekst valide is voor training."""
        if not text or len(text) < min_length:
            return False

        words = text.split()
        if len(words) < min_words:
            return False

        # Filter zeer korte tokens
        avg_word_length = sum(len(w) for w in words) / len(words)
        if avg_word_length < 2:
            return False

        # Filter teksten met te veel getallen
        digit_ratio = sum(1 for c in text if c.isdigit()) / len(text)
        if digit_ratio > 0.3:
            return False

        return True

    def preprocess(self, text, remove_urls=True, remove_emails=True,
                   remove_html=True, keep_punctuation=True):
        """
        Volledig preprocessing pipeline.

        Args:
            text: Input tekst
            remove_urls: Verwijder URLs
            remove_emails: Verwijder emails
            remove_html: Verwijder HTML tags
            keep_punctuation: Behoud basale punctuatie

        Returns:
            Schone tekst
        """
        # 1. Unicode normalisatie
        text = self.normalize_unicode(text)

        # 2. Verwijder control characters
        text = self.remove_control_chars(text)

        # 3. Verwijder HTML
        if remove_html:
            text = self.remove_html(text)

        # 4. Verwijder URLs
        if remove_urls:
            text = self.remove_urls(text)

        # 5. Verwijder emails
        if remove_emails:
            text = self.remove_emails(text)

        # 6. Normaliseer whitespace
        text = self.normalize_whitespace(text)

        # 7. Verwijder special characters
        text = self.remove_special_chars(text, keep_punctuation=keep_punctuation)

        # 8. Normaliseer whitespace opnieuw
        text = self.normalize_whitespace(text)

        return text


def preprocess_dataset(dataset, text_column='text', batch_size=1000):
    """
    Verwerk een volledig dataset.

    Args:
        dataset: Hugging Face dataset
        text_column: Naam van tekst kolom
        batch_size: Batch size voor processing

    Returns:
        Verwerkt dataset
    """
    preprocessor = DutchTextPreprocessor()

    def process_batch(batch):
        """Verwerk batch."""
        processed_texts = []
        valid_indices = []

        for i, text in enumerate(batch[text_column]):
            if text is None:
                continue

            # Preprocessing
            clean_text = preprocessor.preprocess(text)

            # Validatie
            if preprocessor.is_valid_text(clean_text):
                processed_texts.append(clean_text)
                valid_indices.append(i)

        return {
            'text': processed_texts,
            'indices': valid_indices
        }

    logger.info(f"Processing dataset with {len(dataset)} samples...")

    processed_texts = []

    # Process in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Preprocessing"):
        batch = dataset[i:i+batch_size]
        result = process_batch(batch)
        processed_texts.extend(result['text'])

    logger.info(f"Retained {len(processed_texts)} valid samples ({len(processed_texts)/len(dataset)*100:.1f}%)")

    # Create new dataset
    processed_dataset = Dataset.from_dict({'text': processed_texts})
    return processed_dataset


def main():
    """Main preprocessing pipeline."""
    print("=" * 70)
    print("[*] Dutch Text Preprocessing Pipeline")
    print("=" * 70)

    preprocessor = DutchTextPreprocessor()

    # Example texts
    examples = [
        "Dit is een normale Nederlandse zin met normale spelling.",
        "<p>Dit is HTML!</p> Met een URL: https://example.com en email test@example.com",
        "VEEL   SPATIES    HIER    EN    NEWLINES\n\n\nMEER NEWLINES",
        "Cijfers 123456789 en speciale chars: @#$%^&*()",
        "Korte",  # Te kort
    ]

    print("\n[EXAMPLES] Before & After:")
    print("-" * 70)

    for i, text in enumerate(examples, 1):
        clean = preprocessor.preprocess(text)
        valid = preprocessor.is_valid_text(clean)

        print(f"\n{i}. Original:")
        print(f"   {text[:60]}...")
        print(f"   Clean: {clean[:60]}...")
        print(f"   Valid: {'OK' if valid else 'INVALID'}")

    # Process real dataset
    print("\n" + "=" * 70)
    print("[*] Processing Wikipedia NL Dataset...")
    print("=" * 70)

    try:
        wiki_path = Path('E:/Claude/workflow/WatergeusLLM/datasets/wikipedia_nl')
        wiki_data = load_from_disk(str(wiki_path))

        # Handle DatasetDict
        if hasattr(wiki_data, 'keys'):
            wiki_dataset = wiki_data['train']
        else:
            wiki_dataset = wiki_data

        logger.info(f"Original size: {len(wiki_dataset)} samples")

        # Preprocess (sample first 50k for meaningful training data)
        sample_size = min(50000, len(wiki_dataset))
        sample_dataset = wiki_dataset.select(range(sample_size))

        processed = preprocess_dataset(sample_dataset, text_column='text')

        print(f"\n[OK] Preprocessing complete!")
        print(f"  Original: {sample_size} samples")
        print(f"  Processed: {len(processed)} samples")
        print(f"  Filtered: {sample_size - len(processed)} samples")

        # Show some examples
        print(f"\n[EXAMPLES] Processed Wikipedia texts:")
        for i in range(min(3, len(processed))):
            text = processed['text'][i]
            print(f"\n  {i+1}. {text[:100]}...")

        # Save processed sample
        output_path = Path('E:/Claude/workflow/WatergeusLLM/datasets/wikipedia_nl_sample_processed')
        processed.save_to_disk(str(output_path))
        print(f"\n[OK] Sample saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
