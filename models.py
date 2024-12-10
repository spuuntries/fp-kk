import random
from collections import defaultdict, Counter
import re
import numpy as np
from typing import List
from nltk.stem import PorterStemmer
from difflib import SequenceMatcher
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import nltk
from nltk.corpus import brown
from nltk import pos_tag
from deap import base, creator, tools, algorithms
import torch.nn.functional as F
import gc
from tqdm import tqdm
import math


class GAModel:
    def __init__(self):
        if "FitnessMax" not in creator.__dict__:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", list, fitness=creator.FitnessMax)

        nltk.download("brown", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        nltk.download("punkt_tab", quiet=True)

        # Regular dicts instead of defaultdicts
        self.vocab_by_pos = {}
        self.unigram_counts = {}
        self.bigram_counts = {}
        self.trigram_counts = {}
        self.fourgram_counts = {}
        self.pos_patterns = {}
        self.position_counts = {}

        self.suffix_punct = {",", ".", ";", ":", "!", "?"}
        self.prefix_punct = {'"', "(", "[", "{"}
        self.suffix_punct = {'"', ")", "]", "}"}
        self.contractions = {"s", "t", "ll", "re", "ve", "m", "d"}

        self.max_length = 15
        self.min_word_freq = 3
        self.vocab_size = 50000
        self.trained = False

        self.toolbox = base.Toolbox()
        self.toolbox.register("select", tools.selTournament, tournsize=5)

    def get_ngram_count(self, ngram_dict, *keys):
        current = ngram_dict
        for key in keys:
            if key not in current:
                return 0
            current = current[key]
        return current if isinstance(current, int) else 0

    def fit(self, texts: List[str]):
        print("Training on provided texts...")

        # Convert texts to tokenized sentences if they aren't already
        sentences = []
        for text in texts:
            if isinstance(text, str):
                words = nltk.word_tokenize(text)
                if words:
                    sentences.append(words)

        # First pass: count word frequencies
        print("Counting word frequencies...")
        word_freqs = {}
        for sentence in tqdm(sentences):
            for word in sentence:
                word = word.lower()
                word_freqs[word] = word_freqs.get(word, 0) + 1

        self.valid_words = {
            word
            for word, freq in sorted(
                word_freqs.items(), key=lambda x: x[1], reverse=True
            )[: self.vocab_size]
            if freq >= self.min_word_freq
        }

        print(f"Vocab size: {len(self.valid_words)}")

        del word_freqs

        # Second pass: build n-grams
        print("Building n-grams...")
        temp_vocab_by_pos = {}
        temp_unigram_counts = {}
        temp_bigram_counts = {}
        temp_trigram_counts = {}
        temp_fourgram_counts = {}
        temp_pos_patterns = {}

        for sentence in tqdm(sentences):
            if len(sentence) < 5:
                continue

            words = [
                word.lower()
                for word in sentence
                if word.lower() in self.valid_words or word in self.suffix_punct
            ]

            if len(words) < 3:
                continue

            pos_tags = pos_tag(words)

            # Update vocabularies and n-grams
            for i, (word, tag) in enumerate(pos_tags):
                if tag not in temp_vocab_by_pos:
                    temp_vocab_by_pos[tag] = set()
                temp_vocab_by_pos[tag].add(word)
                temp_unigram_counts[word] = temp_unigram_counts.get(word, 0) + 1

                if i < len(words) - 1:
                    w2 = words[i + 1]
                    if word not in temp_bigram_counts:
                        temp_bigram_counts[word] = {}
                    temp_bigram_counts[word][w2] = (
                        temp_bigram_counts[word].get(w2, 0) + 1
                    )

                    if i < len(words) - 2:
                        w3 = words[i + 2]
                        if word not in temp_trigram_counts:
                            temp_trigram_counts[word] = {}
                        if w2 not in temp_trigram_counts[word]:
                            temp_trigram_counts[word][w2] = {}
                        temp_trigram_counts[word][w2][w3] = (
                            temp_trigram_counts[word][w2].get(w3, 0) + 1
                        )

            # Learn POS patterns
            if len(pos_tags) >= 4:
                for i in range(len(pos_tags) - 3):
                    pattern = tuple(tag for _, tag in pos_tags[i : i + 4])
                    if pattern[0] not in temp_pos_patterns:
                        temp_pos_patterns[pattern[0]] = []
                    temp_pos_patterns[pattern[0]].append(pattern[1:])

        # Convert temporary structures to final form
        self.vocab_by_pos = {
            tag: list(words) for tag, words in temp_vocab_by_pos.items()
        }
        self.unigram_counts = temp_unigram_counts
        self.bigram_counts = temp_bigram_counts
        self.trigram_counts = temp_trigram_counts
        self.fourgram_counts = temp_fourgram_counts
        self.pos_patterns = temp_pos_patterns

        del temp_vocab_by_pos, temp_unigram_counts, temp_bigram_counts
        del temp_trigram_counts, temp_fourgram_counts, temp_pos_patterns
        gc.collect()

        self.trained = True
        print("Training complete!")
        return self

    def clean_text(self, text: str) -> str:
        words = text.split()
        cleaned = []

        for word in words:
            if not word:
                continue

            if word in self.suffix_punct:
                if cleaned:
                    cleaned[-1] = cleaned[-1] + word
            elif word.startswith(",") or word.startswith("."):
                if cleaned:
                    cleaned[-1] = cleaned[-1] + word[0]
                if len(word) > 1:
                    cleaned.append(word[1:])
            else:
                cleaned.append(word)

        result = " ".join(cleaned)

        for punct in self.suffix_punct:
            while f"{punct} {punct}" in result:
                result = result.replace(f"{punct} {punct}", punct)

        while "  " in result:
            result = result.replace("  ", " ")

        return result.strip()

    def get_likely_word(self, pos: str, context: List[str], tau: float = 1.0):
        candidates = self.vocab_by_pos.get(pos, [])
        if not candidates:
            return random.choice(
                [
                    w
                    for w in self.valid_words
                    if w not in {"of", "the", "and", "in", "to", "a", "is", "was"}
                ]
            )

        # Prioritize content words
        content_candidates = [
            w
            for w in candidates
            if w not in {"of", "the", "and", "in", "to", "a", "is", "was"}
        ]

        if content_candidates:
            candidates = content_candidates

        # Calculate probabilities for candidates
        scores = []
        context = context[-3:] if len(context) > 3 else context

        # Get raw scores
        for word in candidates:
            score = self.unigram_counts.get(word, 0) + 1
            if len(context) >= 1:
                score *= self.get_ngram_count(self.bigram_counts, context[-1], word) + 1
            if len(context) >= 2:
                score *= (
                    self.get_ngram_count(
                        self.trigram_counts, context[-2], context[-1], word
                    )
                    + 1
                )
            scores.append(score)

        # Convert to probabilities
        scores = np.array(scores)
        probs = scores / scores.sum()

        # Sort probabilities in descending order
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        # Estimate s (slope parameter)
        def estimate_s(probs):
            result = 0
            num = 0
            den = 0
            for i in range(min(len(probs) - 1, 99)):
                b = probs[i] / probs[i + 1]
                t = (i + 2) / (i + 1)
                num += math.log(b) * math.log(t)
                den += math.log(t) ** 2
            return num / den if den != 0 else 1.0

        # Compute k (vocabulary cutoff)
        def compute_k(n, s, tau):
            eps = s - 1
            if eps == 0 or n <= 1:
                return n
            k = ((eps * (2**tau)) / (1 - n ** (-eps))) ** (1 / s)
            return max(1, min(n, round(k)))

        s = estimate_s(sorted_probs)
        k = compute_k(len(candidates), s, tau)

        # Truncate to top-k
        sorted_probs = sorted_probs[:k]
        sorted_indices = sorted_indices[:k]

        # Renormalize probabilities
        sorted_probs = sorted_probs / sorted_probs.sum()

        # Sample from truncated distribution
        chosen_idx = np.random.choice(k, p=sorted_probs)
        chosen_word = candidates[sorted_indices[chosen_idx]]

        return chosen_word

    def find_similar_pattern(self, input_sentence: str) -> List[str]:
        words = nltk.word_tokenize(input_sentence.lower())
        pos_tags = pos_tag(words)

        if len(pos_tags) >= 1:
            last_pos = pos_tags[-1][1]
            if last_pos in self.pos_patterns:
                return list(random.choice(self.pos_patterns[last_pos]))

        return ["DT", "NN", "VB"]

    def mutate_word(self, individual):
        """More aggressive mutation that replaces problematic sequences"""
        if len(individual) < 3:
            return (individual,)

        # Find problematic sequences
        for i in range(len(individual) - 2):
            if (
                individual[i : i + 3].count("of") > 1
                or individual[i : i + 3].count("the") > 1
            ):
                try:
                    pos_tags = nltk.pos_tag(individual[i : i + 3])
                    for j, (word, tag) in enumerate(pos_tags):
                        # Replace with words that aren't common function words
                        new_word = self.get_likely_word(
                            tag, individual[max(0, i - 1) : i + j]
                        )
                        if new_word not in {"of", "the", "and", "in", "to"}:
                            individual[i + j] = new_word
                except Exception:
                    pass

        return (individual,)

    def generate(
        self,
        input_sentence: str,
        target_length: int = None,
        population_size=50,
        generations=20,
    ) -> str:
        if "FitnessMax" in creator.__dict__:
            del creator.FitnessMax
        if "Individual" in creator.__dict__:
            del creator.Individual

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        if target_length is None:
            target_length = min(len(input_sentence.split()) + self.max_length, 30)

        input_words = input_sentence.lower().split()
        current_length = len(input_words)
        words_to_generate = max(2, target_length - current_length)

        # Create fresh toolbox for each generation
        toolbox = base.Toolbox()

        def create_individual():
            result = []
            context = input_words[-3:] if len(input_words) > 3 else input_words

            while len(result) < words_to_generate:
                pattern = self.find_similar_pattern(" ".join(context))
                word = self.get_likely_word(pattern[0], context)
                result.append(word)
                context = (context + [word])[-3:]

                if word in {".", "!", "?"}:
                    break

            return creator.Individual(result)

        def evaluate(individual):
            score = 0
            context = input_words[-3:] if len(input_words) > 3 else input_words

            # Break into chunks of 3-4 words and evaluate their naturalness
            chunks = []
            current_chunk = []

            for word in individual:
                current_chunk.append(word)
                if len(current_chunk) >= 3:
                    if self.get_ngram_count(self.trigram_counts, *current_chunk) > 0:
                        score += 10  # Reward natural 3-word sequences heavily
                        chunks.append(current_chunk)
                        current_chunk = []

            # Penalize harshly for repetitive phrases
            seen_phrases = set()
            for i in range(len(individual) - 2):
                phrase = " ".join(individual[i : i + 3])
                if phrase in seen_phrases:
                    score -= 20  # Heavy penalty for repetition
                seen_phrases.add(phrase)

            # Strict limits on common patterns
            of_count = individual.count("of")
            the_count = individual.count("the")
            and_count = individual.count("and")

            if of_count > 1:
                score -= (of_count - 1) * 15
            if the_count > 2:
                score -= (the_count - 2) * 15
            if and_count > 1:
                score -= (and_count - 1) * 15

            return (score,)

        if words_to_generate <= 1:
            # If we need very few words, just use simple generation instead of GA
            context = input_words[-3:] if len(input_words) > 3 else input_words
            pattern = self.find_similar_pattern(" ".join(context))
            result = input_sentence + " " + self.get_likely_word(pattern[0], context)
            return self.clean_text(result)

        # Modify mate operation to handle small sequences
        def safe_mate(ind1, ind2):
            if len(ind1) < 2 or len(ind2) < 2:
                return ind1, ind2
            return tools.cxTwoPoint(ind1, ind2)

        # Register the safe mate function instead
        toolbox.register("mate", safe_mate)

        # Register all functions to fresh toolbox
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mutate", self.mutate_word)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Create and evolve population
        pop = toolbox.population(n=population_size)
        algorithms.eaSimple(
            pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=generations, verbose=False
        )

        best_individual = tools.selBest(pop, 1)[0]
        result = input_sentence + " " + " ".join(best_individual)
        result = self.clean_text(result)

        del creator.FitnessMax
        del creator.Individual

        return result


class MarkovModel:
    def __init__(self, order=4):
        self.order = order
        self.word_chain = defaultdict(lambda: defaultdict(int))
        self.char_chain = defaultdict(lambda: defaultdict(int))
        self.vocabulary = set()
        self.word_counts = defaultdict(int)
        self.unknown_token = "<UNK>"
        self.start_token = "<START>"
        self.end_token = "<END>"

    def reset(self):
        self.word_chain = defaultdict(lambda: defaultdict(int))
        self.char_chain = defaultdict(lambda: defaultdict(int))
        self.vocabulary = set()
        self.word_counts = defaultdict(int)
        self.word_chain_probs = defaultdict(dict)
        self.char_chain_probs = defaultdict(dict)

    def preprocess_text(self, text: str) -> List[str]:
        text = " ".join(text.lower().split())
        words = [word.strip() for word in re.findall(r"\b\w+\b", text)]
        return words if words else []

    def predict_next(self, seed_sequence: List[str], temperature=1.0):
        """Predict next word using backoff strategy"""
        # Start with full order (4-gram) and back off to smaller n-grams
        for n in range(self.order, 0, -1):
            # Get the context of size n
            context = tuple(seed_sequence[-n:])

            if context in self.word_chain_probs:
                probs = self.word_chain_probs[context]
                if temperature != 1.0:
                    probs = self._apply_temperature(probs, temperature)
                return random.choices(list(probs.keys()), list(probs.values()))[0]

        # If no n-gram matches found, fallback to random
        return random.choice(list(self.vocabulary))

    def fit(self, texts: List[str]):
        """Modified fit to store all n-gram orders"""
        for text in tqdm(texts):
            words = self.preprocess_text(text)
            if not words:
                continue

            # Add words to vocabulary
            for word in words:
                self.vocabulary.add(word)
                self.word_counts[word] += 1

            # Store n-grams of all orders up to self.order
            padded_words = [self.start_token] * self.order + words + [self.end_token]
            for n in range(1, self.order + 1):
                for i in range(len(padded_words) - n):
                    context = tuple(padded_words[i : i + n])
                    next_word = padded_words[i + n]
                    self.word_chain[context][next_word] += 1

        self._normalize_probabilities()
        return self

    def _normalize_probabilities(self):
        self.word_chain_probs = defaultdict(dict)
        self.char_chain_probs = defaultdict(dict)

        for context in self.word_chain:
            total = sum(self.word_chain[context].values())
            for word in self.word_chain[context]:
                self.word_chain_probs[context][word] = (
                    self.word_chain[context][word] / total
                )

        for char in self.char_chain:
            total = sum(self.char_chain[char].values())
            for next_char in self.char_chain[char]:
                self.char_chain_probs[char][next_char] = (
                    self.char_chain[char][next_char] / total
                )

    def generate_text(self, num_words=50, temperature=1.0, seed_sequence=None):
        """
        Generate text with optional seed sequence.

        Args:
            num_words: Number of words to generate
            temperature: Sampling temperature
            seed_sequence: Optional list or string of seed text
        """
        # Handle string input for seed sequence
        if isinstance(seed_sequence, str):
            seed_sequence = self.preprocess_text(seed_sequence)
        elif seed_sequence is None:
            seed_sequence = []

        # Process seed sequence
        if seed_sequence:
            # If seed sequence is shorter than order, pad with start tokens
            if len(seed_sequence) < self.order:
                current_words = [self.start_token] * (
                    self.order - len(seed_sequence)
                ) + seed_sequence
            else:
                # Take the last 'order' words
                current_words = seed_sequence[-self.order :]

            # Initialize generated_words with the seed sequence
            generated_words = seed_sequence[:]
        else:
            current_words = [self.start_token] * self.order
            generated_words = []

        for _ in range(num_words):
            next_word = self.predict_next(current_words, temperature)

            if next_word == self.end_token:
                if len(generated_words) < (num_words + len(seed_sequence)) // 2:
                    current_words = [self.start_token] * self.order
                    continue
                else:
                    break

            generated_words.append(next_word)
            current_words.append(next_word)
            current_words = current_words[1:]

        return " ".join(generated_words)

    def _apply_temperature(self, probabilities, temperature):
        probs = np.array(list(probabilities.values()))
        probs = np.log(probs + 1e-10) / temperature
        probs = np.exp(probs)
        probs = probs / np.sum(probs)
        return dict(zip(probabilities.keys(), probs))


class SelfAttentionModel(nn.Module):
    def __init__(self, vocab_size, embed_size=128, num_heads=8, num_layers=3):
        super(SelfAttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(embed_size, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        return self.fc(x)

    def calculate_perplexity(self, dataloader):
        self.eval()
        device = next(self.parameters()).device
        total_log_likelihood = 0
        total_words = 0

        with torch.no_grad():
            for input_seq, target_seq in dataloader:
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)

                logits = self(input_seq)
                log_probs = F.log_softmax(logits, dim=-1)

                # Calculate log likelihood for non-padding tokens
                mask = target_seq != 0  # Assuming 0 is padding
                log_likelihood = torch.gather(
                    log_probs, dim=-1, index=target_seq.unsqueeze(-1)
                ).squeeze(-1)

                total_log_likelihood += log_likelihood[mask].sum().item()
                total_words += mask.sum().item()

        if total_words == 0:
            return float("inf")

        return np.exp(-total_log_likelihood / total_words)

    def generate(
        self, prompt: str, max_length: int = 50, temperature: float = 1.0
    ) -> str:
        """
        Generate text continuation from a prompt

        Args:
            prompt: Input text to continue
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature (higher = more diverse)
        """
        self.eval()
        device = next(self.parameters()).device

        # Convert prompt to indices
        tokens = ["<START>"] + prompt.lower().split()
        input_ids = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in tokens]
        input_ids = (
            torch.tensor(input_ids).unsqueeze(0).to(device)
        )  # Add batch dimension

        generated_words = tokens[1:]  # Remove <START> token

        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                outputs = self(input_ids)
                next_token_logits = outputs[0, -1, :] / temperature

                # Apply softmax to get probabilities
                probs = torch.softmax(next_token_logits, dim=-1)

                # Sample from the probability distribution
                next_token = torch.multinomial(probs, num_samples=1)

                # Convert to word and add to generated sequence
                next_word = self.idx2word[next_token.item()]

                # Stop if we generate END token or reach max length
                if next_word == "<END>":
                    break

                generated_words.append(next_word)

                # Update input_ids for next iteration
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        return " ".join(generated_words)

    def set_vocab_mappings(self, word2idx, idx2word):
        """Set vocabulary mappings for generation"""
        self.word2idx = word2idx
        self.idx2word = idx2word


class LanguageModelDataset(Dataset):
    def __init__(self, sentences, word2idx, max_len=50):
        self.sentences = sentences
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = (
            ["<START>"] + self.sentences[idx].split()[: self.max_len - 2] + ["<END>"]
        )
        input_seq = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words[:-1]]
        target_seq = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words[1:]]

        input_seq += [self.word2idx["<PAD>"]] * (self.max_len - len(input_seq))
        target_seq += [self.word2idx["<PAD>"]] * (self.max_len - len(target_seq))

        return torch.tensor(input_seq), torch.tensor(target_seq)
