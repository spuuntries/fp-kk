import time
import psutil
import torch
import numpy as np
import math
import tracemalloc
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List, Dict, Any
from collections import Counter
from models import LanguageModelDataset


def measure_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def calculate_perplexity(model: Any, test_data: List[str]) -> float:
    """
    Calculate perplexity for any of the three models

    PPL = exp(-1/N * sum(log P(w_i|context)))
    """
    if isinstance(model, torch.nn.Module):  # Self-Attention Model
        return model.calculate_perplexity(test_data)  # test_data is a DataLoader here

    total_log_likelihood = 0
    total_words = 0

    if hasattr(model, "get_ngram_count"):  # ContinuationGenerator
        for sentence in test_data:
            words = sentence.lower().split()
            for i in range(len(words) - 2):
                context = words[i : i + 2]
                next_word = words[i + 2]

                # Get counts with proper backoff
                if (
                    context[0] in model.trigram_counts
                    and context[1] in model.trigram_counts[context[0]]
                ):
                    # Trigram probability with interpolation
                    trigram_count = model.get_ngram_count(
                        model.trigram_counts, context[0], context[1], next_word
                    )
                    context_count = sum(
                        model.trigram_counts[context[0]][context[1]].values()
                    )

                    # Interpolate with bigram and unigram probabilities
                    trigram_prob = (trigram_count + 1) / (
                        context_count + len(model.valid_words)
                    )
                    bigram_prob = (
                        model.get_ngram_count(
                            model.bigram_counts, context[1], next_word
                        )
                        + 1
                    ) / (
                        sum(model.bigram_counts.get(context[1], {}).values())
                        + len(model.valid_words)
                    )
                    unigram_prob = (model.unigram_counts.get(next_word, 0) + 1) / (
                        sum(model.unigram_counts.values()) + len(model.valid_words)
                    )

                    # Interpolation weights
                    lambda1, lambda2, lambda3 = 0.7, 0.2, 0.1
                    prob = (
                        lambda1 * trigram_prob
                        + lambda2 * bigram_prob
                        + lambda3 * unigram_prob
                    )

                    total_log_likelihood += np.log(prob)
                    total_words += 1

    elif hasattr(model, "word_chain_probs"):  # Markov Model
        for sentence in test_data:
            words = model.preprocess_text(sentence)
            for i in range(len(words) - model.order):
                context = tuple(words[i : i + model.order])
                next_word = words[i + model.order]

                # Use interpolation between n-gram orders
                probs = []
                total_lambda = 0

                # Full context probability
                if context in model.word_chain:
                    count = model.word_chain[context].get(next_word, 0)
                    total = sum(model.word_chain[context].values())
                    probs.append((0.7, (count + 1) / (total + len(model.vocabulary))))
                    total_lambda += 0.7

                # Lower order context
                if len(context) > 1:
                    shorter_context = context[1:]
                    if shorter_context in model.word_chain:
                        count = model.word_chain[shorter_context].get(next_word, 0)
                        total = sum(model.word_chain[shorter_context].values())
                        probs.append(
                            (0.2, (count + 1) / (total + len(model.vocabulary)))
                        )
                        total_lambda += 0.2

                # Unigram probability
                unigram_count = sum(1 for w in model.word_counts if w == next_word)
                total_words_count = sum(model.word_counts.values())
                probs.append(
                    (
                        0.1,
                        (unigram_count + 1)
                        / (total_words_count + len(model.vocabulary)),
                    )
                )
                total_lambda += 0.1

                # Combine probabilities
                prob = sum(weight * p for weight, p in probs) / total_lambda
                total_log_likelihood += np.log(prob)
                total_words += 1

    # Calculate final perplexity
    if total_words == 0:
        return float("inf")

    avg_log_likelihood = total_log_likelihood / total_words
    perplexity = np.exp(-avg_log_likelihood)

    # Sanity check
    if perplexity > 100000:
        print(
            f"Warning: Very high perplexity ({perplexity}). This might indicate a calculation error."
        )

    return perplexity


def prepare_data(texts: List[str], vocab_size: int = 10000):
    """
    Prepare data for all models, including vocabulary creation for the transformer
    """
    # Create vocabulary
    word_counts = Counter()
    for text in texts:
        words = text.lower().split()
        word_counts.update(words)

    # Create word2idx mapping
    special_tokens = ["<PAD>", "<UNK>", "<START>", "<END>"]
    vocab = special_tokens + [
        word for word, _ in word_counts.most_common(vocab_size - len(special_tokens))
    ]
    word2idx = {word: idx for idx, word in enumerate(vocab)}

    return {"texts": texts, "word2idx": word2idx, "vocab_size": len(vocab)}


def train_self_attention(model, train_loader, val_loader, num_epochs=10, patience=3):
    """
    Train the self-attention model with early stopping
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # 0 is pad_idx
    optimizer = torch.optim.Adam(model.parameters())

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in tqdm(range(num_epochs)):
        # Training
        model.train()
        total_loss = 0
        for input_seq, target_seq in train_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)

            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output.view(-1, output.size(-1)), target_seq.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_seq, target_seq in val_loader:
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)
                output = model(input_seq)
                val_loss += criterion(
                    output.view(-1, output.size(-1)), target_seq.view(-1)
                ).item()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    return model


def compare_models(
    train_texts: List[str],
    test_texts: List[str],
    models: Dict[str, Any],
    train_loader=None,
    val_loader=None,
    test_loader=None,
) -> Dict[str, Dict[str, float]]:
    results = {}

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        # Measure training time and memory
        tracemalloc.start()
        start_time = time.time()
        start_memory = measure_memory_usage()

        # Train model
        if isinstance(model, torch.nn.Module):
            model = train_self_attention(model, train_loader, val_loader)
        else:
            model.fit(train_texts)

        training_time = time.time() - start_time
        memory_used = measure_memory_usage() - start_memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Measure generation/evaluation time and calculate perplexity
        start_time = time.time()

        # Calculate perplexity
        if isinstance(model, torch.nn.Module):
            perplexity = calculate_perplexity(model, test_loader)
        else:
            perplexity = calculate_perplexity(model, test_texts)

        # Validate perplexity
        if not (1 < perplexity < 100000):
            print(f"Warning: Unusual perplexity value ({perplexity}) for {name}")
            print("Running additional validation...")

            # Try with a small subset for validation
            small_test = (
                test_texts[:10] if not isinstance(test_texts, DataLoader) else None
            )
            if small_test:
                validation_ppl = calculate_perplexity(model, small_test)
                print(f"Validation perplexity on small subset: {validation_ppl}")

        # Generate samples for timing
        for _ in range(5):
            if isinstance(model, torch.nn.Module):
                model.generate("The cat sat on")
            elif hasattr(model, "generate"):
                model.generate("The cat sat on")
            else:
                model.generate_text(num_words=20, seed_sequence="The cat sat on")

        generation_time = (time.time() - start_time) / 5

        results[name] = {
            "Training Time (s)": training_time,
            "Generation Time (s)": generation_time,
            "Memory Usage (MB)": memory_used,
            "Peak Memory (MB)": peak / 1024 / 1024,
            "Perplexity": perplexity,
        }

    return results


def print_results(results: Dict[str, Dict[str, float]]):
    """
    Print comparison results in a formatted table
    """
    metrics = [
        "Training Time (s)",
        "Generation Time (s)",
        "Memory Usage (MB)",
        "Peak Memory (MB)",
        "Perplexity",
    ]

    # Print header
    print("\nModel Comparison Results:")
    print("-" * 100)
    print(f"{'Model':<20}", end="")
    for metric in metrics:
        print(f"{metric:<20}", end="")
    print()
    print("-" * 100)

    # Print results
    for model_name, metrics_dict in results.items():
        print(f"{model_name:<20}", end="")
        for metric in metrics:
            value = metrics_dict[metric]
            print(f"{value:<20.4f}", end="")
        print()
    print("-" * 100)
