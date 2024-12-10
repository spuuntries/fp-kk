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
from models import (
    GAModel,
    MarkovModel,
    SelfAttentionModel,
    LanguageModelDataset,
)


def measure_memory_usage():
    """Get current memory usage in MB using multiple measurements"""
    process = psutil.Process()
    measurements = []
    for _ in range(5):  # Take several measurements
        measurements.append(process.memory_info().rss / 1024 / 1024)
        time.sleep(0.1)  # Short delay between measurements
    return np.median(measurements)  # Return median value


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
        # Pre-compute constants
        vocab_size = len(model.vocabulary)
        total_word_count = sum(model.word_counts.values())

        # Cache lambda weights
        lambda1, lambda2, lambda3 = 0.7, 0.2, 0.1

        for sentence in test_data:
            words = model.preprocess_text(sentence)
            if len(words) < model.order + 1:
                continue

            padded_words = [model.start_token] * model.order + words

            for i in range(len(words)):
                context = tuple(padded_words[i : i + model.order])
                next_word = words[i]

                prob = 0

                # Full context probability (highest weight)
                if context in model.word_chain_probs:
                    context_probs = model.word_chain_probs[context]
                    if next_word in context_probs:
                        prob += lambda1 * context_probs[next_word]

                # Backoff to lower order (only if needed)
                if (
                    prob < 0.5 and len(context) > 1
                ):  # Only backoff if high-order prob is low
                    shorter_context = context[1:]
                    if shorter_context in model.word_chain_probs:
                        context_probs = model.word_chain_probs[shorter_context]
                        if next_word in context_probs:
                            prob += lambda2 * context_probs[next_word]

                # Unigram probability (only if needed)
                if prob < 0.1:  # Only use unigram if higher-order probs are very low
                    unigram_prob = (
                        model.word_counts.get(next_word, 0) / total_word_count
                    )
                    prob += lambda3 * (unigram_prob + 1 / vocab_size)

                # Ensure prob is not zero (smoothing)
                prob = max(prob, 1e-10)

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


def train_self_attention(model, train_loader, val_loader, num_epochs=100, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # 0 is pad_idx
    optimizer = torch.optim.Adam(model.parameters())

    # Add ReduceLROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    pbar = tqdm(range(num_epochs), desc="Training")

    for epoch in pbar:
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

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Update progress bar with learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_description(f"Training (lr={current_lr:.2e})")

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

    results["Models"] = models
    return results


def compare_models_with_stats(
    train_texts: List[str],
    test_texts: List[str],
    word2idx=None,
    idx2word=None,
    train_loader=None,
    val_loader=None,
    test_loader=None,
    num_runs: int = 20,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    models = {
        "Genetic Algorithm": GAModel(),
        "Markov Model": MarkovModel(order=2),
        "Self-Attention": SelfAttentionModel(
            vocab_size=len(word2idx), embed_size=128, num_heads=8, num_layers=4
        ),
    }

    # Set vocabulary mappings for transformer model
    models["Self-Attention"].set_vocab_mappings(word2idx, idx2word)

    all_results = {
        name: {
            "Training Time (s)": [],
            "Generation Time (s)": [],
            "Memory Usage (MB)": [],
            "Peak Memory (MB)": [],
            "Perplexity": [],
        }
        for name in models.keys()
    }

    # Store individual run results
    run_results = []

    # Run multiple times
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")

        results = compare_models(
            train_texts, test_texts, models, train_loader, val_loader, test_loader
        )

        run_results.append(results)

        # Collect results from each run
        for model_name, metrics in results.items():
            if model_name != "Models":
                for metric, value in metrics.items():
                    all_results[model_name][metric].append(value)

    # Calculate statistics
    final_results = {}
    for model_name, metrics in all_results.items():
        final_results[model_name] = {
            metric: {"mean": np.mean(values), "std": np.std(values), "values": values}
            for metric, values in metrics.items()
        }
    final_results["Models"] = results["Models"]

    return final_results


def print_results_with_stats(results: Dict[str, Dict[str, Dict[str, float]]]):
    """
    Print comparison results with individual runs and statistics
    """
    metrics = [
        "Training Time (s)",
        "Generation Time (s)",
        "Memory Usage (MB)",
        "Peak Memory (MB)",
        "Perplexity",
    ]

    for model_name, metrics_dict in results.items():
        if model_name != "Models":
            print(f"\n{'-' * 80}")
            print(f"Results for {model_name}:")
            print(f"{'-' * 80}")

            for metric in metrics:
                values = metrics_dict[metric]["values"]
                mean = metrics_dict[metric]["mean"]
                std = metrics_dict[metric]["std"]

                print(f"\n{metric}:")
                print(f"Individual runs: ", end="")
                for val in values:
                    print(f"{val:.4f}, ", end="")
                print(f"\nMean ± Std: {mean:.4f} ± {std:.4f}")

    # Print summary table
    print(f"\n{'-' * 140}")
    print("Summary Table:")
    print(f"{'-' * 140}")
    print(f"{'Model':<20}", end="")
    for metric in metrics:
        print(f"{metric:<24}", end="")
    print()
    print(f"{'-' * 140}")

    for model_name, metrics_dict in results.items():
        if model_name != "Models":
            print(f"{model_name:<20}", end="")
            for metric in metrics:
                mean = metrics_dict[metric]["mean"]
                std = metrics_dict[metric]["std"]
                print(f"{mean:>.2f} ± {std:>.2f}".ljust(24), end="")
            print()
    print(f"{'-' * 140}")
