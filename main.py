import nltk
from nltk.corpus import brown
from models import (
    ContinuationGenerator,
    AdvancedMarkovModel,
    SelfAttentionModel,
    LanguageModelDataset,
)
from evaluation import compare_models, print_results, prepare_data
import torch
from torch.utils.data import random_split, DataLoader
import random
import numpy as np


def set_seeds(seed=42):
    """Set all seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_brown_corpus(num_sentences=None):
    """Prepare Brown corpus data"""
    # Download required NLTK data
    nltk.download("brown", quiet=True)

    # Get sentences and convert to texts
    sentences = brown.sents()
    if num_sentences:
        sentences = sentences[:num_sentences]

    texts = [" ".join(sent).lower() for sent in sentences]

    # Split into train and test
    train_size = int(0.8 * len(texts))
    train_texts = texts[:train_size]
    test_texts = texts[train_size:]

    return train_texts, test_texts


def main():
    # Set seeds for reproducibility
    set_seeds(42)

    # Prepare data
    print("Preparing data...")
    train_texts, test_texts = prepare_brown_corpus(num_sentences=1000)

    # Prepare vocabulary and word mappings
    data = prepare_data(train_texts)
    word2idx = data["word2idx"]
    idx2word = {idx: word for word, idx in word2idx.items()}

    # Create datasets and dataloaders for transformer
    train_dataset = LanguageModelDataset(train_texts, word2idx)
    val_dataset = LanguageModelDataset(test_texts[: len(test_texts) // 2], word2idx)
    test_dataset = LanguageModelDataset(test_texts[len(test_texts) // 2 :], word2idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Initialize models
    print("Initializing models...")
    models = {
        "Continuation Generator": ContinuationGenerator(),
        "Markov Model": AdvancedMarkovModel(order=2),
        "Self-Attention": SelfAttentionModel(
            vocab_size=len(word2idx), embed_size=128, num_heads=8, num_layers=3
        ),
    }

    # Set vocabulary mappings for transformer model
    models["Self-Attention"].set_vocab_mappings(word2idx, idx2word)

    # Compare models
    print("Starting model comparison...")
    results = compare_models(
        train_texts=train_texts,
        test_texts=test_texts,
        models=models,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    # Print results
    print_results(results)

    # Generate example outputs
    print("\nExample Generations:")
    print("-" * 50)

    test_prompts = [
        "The cat sat on",
        "In the morning",
        "She walked through",
        "The students were",
        "He looked at the",
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        for name, model in models.items():
            try:
                if isinstance(model, torch.nn.Module):
                    output = model.generate(prompt, max_length=20)
                elif hasattr(model, "generate"):
                    output = model.generate(prompt, target_length=20)
                else:
                    output = model.generate_text(num_words=20)
                print(f"{name}: {output}")
            except Exception as e:
                print(f"{name}: Error generating - {str(e)}")
        print("-" * 50)


if __name__ == "__main__":
    main()