"""
Train Skip-gram with Negative Sampling on a Wikipedia article,
then evaluate the embedding model with intrinsic tests and custom test sets.

Updated experiment:
    - window size changed from 5 to 10

Requirements:
    pip install requests beautifulsoup4 nltk gensim scikit-learn scipy

Optional:
    python -m nltk.downloader punkt stopwords
"""

import re
import json
import random
from typing import List, Tuple, Dict

import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np


WIKI_URL = "https://en.wikipedia.org/wiki/AI_winter"
RANDOM_SEED = 42
WINDOW_SIZE = 10
OUTPUT_MODEL_PATH = "exercise_5_skipgram_sgns_window10.model"
PCA_OUTPUT_PATH = "pca_window10.png"
PCA_WORD_COUNT = 20

# Baseline metrics from the previous model (window=5) to compare against this run.
BASELINE_RESULTS = {
    "window": 5,
    "relatedness_coverage": (9, 12),
    "analogy_coverage": (4, 4),
    "analogy_top5_accuracy": 0.0,
    "direct_similarity": {
        "ai-intelligence": 0.1385,
        "machine-learning": 0.3480,
        "expert-systems": 0.5096,
        "ai-kitchen": None,
    },
}


def ensure_nltk():
    resources = ["punkt", "punkt_tab"]
    for r in resources:
        try:
            nltk.data.find(f"tokenizers/{r}")
        except LookupError:
            nltk.download(r)


def fetch_wikipedia_article(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SGNS-AIWinter-Training/1.0)"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    content_div = soup.find("div", {"id": "mw-content-text"})
    if content_div is None:
        raise ValueError("Could not find Wikipedia article content.")

    paragraphs = content_div.find_all(["p", "li"])
    text_blocks = []

    for p in paragraphs:
        txt = p.get_text(" ", strip=True)
        if txt:
            text_blocks.append(txt)

    text = "\n".join(text_blocks)
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str) -> List[List[str]]:
    sentences = sent_tokenize(text)

    processed = []
    for sent in sentences:
        sent = sent.lower()
        sent = re.sub(r"[^a-z0-9\-\s]", " ", sent)
        sent = re.sub(r"\s+", " ", sent).strip()
        if not sent:
            continue

        tokens = word_tokenize(sent)

        cleaned = []
        for tok in tokens:
            tok = tok.strip("-")
            if not tok:
                continue
            if tok.isdigit():
                continue
            if len(tok) < 2:
                continue
            cleaned.append(tok)

        if len(cleaned) >= 3:
            processed.append(cleaned)

    return processed


def corpus_stats(sentences: List[List[str]]) -> Dict[str, int]:
    flat = [w for s in sentences for w in s]
    vocab = set(flat)
    return {
        "num_sentences": len(sentences),
        "num_tokens": len(flat),
        "vocab_size": len(vocab),
    }


def train_sgns(sentences: List[List[str]]) -> Word2Vec:
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=WINDOW_SIZE,
        min_count=1,
        workers=4,
        sg=1,
        negative=10,
        epochs=200,
        sample=1e-3,
        alpha=0.025,
        min_alpha=0.0007,
        seed=RANDOM_SEED,
    )
    return model


def has_word(model: Word2Vec, word: str) -> bool:
    return word in model.wv.key_to_index


def cosine(model: Word2Vec, w1: str, w2: str) -> float:
    v1 = model.wv[w1].reshape(1, -1)
    v2 = model.wv[w2].reshape(1, -1)
    return float(cosine_similarity(v1, v2)[0][0])


def evaluate_relatedness(model: Word2Vec, test_pairs: List[Tuple[str, str, float]]):
    covered = []

    for w1, w2, score in test_pairs:
        if has_word(model, w1) and has_word(model, w2):
            sim = cosine(model, w1, w2)
            covered.append((w1, w2, score, sim))

    return {
        "covered_items": covered,
        "coverage": len(covered),
        "total": len(test_pairs),
    }


def evaluate_analogies(model: Word2Vec, analogies: List[Tuple[str, str, str, str]]):
    covered = 0
    correct = 0
    details = []

    for a, b, c, d in analogies:
        if all(has_word(model, w) for w in [a, b, c, d]):
            covered += 1
            try:
                preds = model.wv.most_similar(positive=[b, c], negative=[a], topn=5)
                predicted_words = [w for w, _ in preds]
                hit = d in predicted_words
                correct += int(hit)
                details.append(
                    {
                        "analogy": f"{a}:{b}::{c}:?",
                        "expected": d,
                        "predictions": predicted_words,
                        "correct_in_top5": hit,
                    }
                )
            except KeyError:
                pass

    accuracy = correct / covered if covered else float("nan")
    return {
        "coverage": covered,
        "total": len(analogies),
        "accuracy_top5": accuracy,
        "details": details,
    }


def print_top_neighbors(model: Word2Vec, words: List[str], topn: int = 8):
    print("\n=== Nearest Neighbors ===")
    for word in words:
        if has_word(model, word):
            neighbors = model.wv.most_similar(word, topn=topn)
            print(f"\n{word}:")
            for neigh, score in neighbors:
                print(f"  {neigh:20s} {score:.4f}")
        else:
            print(f"\n{word}: [OOV]")


def select_pca_words(model: Word2Vec, word_count: int = 20) -> List[str]:
    # Prefer domain-relevant words first, then backfill from vocab if needed.
    candidate_words = [
        "ai",
        "winter",
        "research",
        "funding",
        "expert",
        "systems",
        "neural",
        "network",
        "machine",
        "learning",
        "artificial",
        "intelligence",
        "symbolic",
        "reasoning",
        "deep",
        "transformer",
        "model",
        "data",
        "algorithm",
        "knowledge",
        "inference",
        "rules",
        "language",
        "bayesian",
        "probability",
    ]

    chosen = [w for w in candidate_words if has_word(model, w)]
    if len(chosen) < word_count:
        for w in model.wv.index_to_key:
            if w not in chosen:
                chosen.append(w)
            if len(chosen) >= word_count:
                break

    return chosen[:word_count]


def plot_pca_embeddings(model: Word2Vec, words: List[str], output_path: str):
    if len(words) < 2:
        raise ValueError("Need at least 2 words for PCA visualization.")

    vectors = np.array([model.wv[w] for w in words])
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(vectors)

    plt.figure(figsize=(13, 9))
    plt.scatter(coords[:, 0], coords[:, 1], s=50, c="#1f77b4")
    for i, word in enumerate(words):
        plt.annotate(word, (coords[i, 0], coords[i, 1]), textcoords="offset points", xytext=(4, 3), fontsize=9)

    plt.title(f"PCA of Word2Vec Embeddings (window={WINDOW_SIZE})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()

    explained = pca.explained_variance_ratio_
    print(f"\nSaved PCA plot to: {output_path}")
    print(f"PCA explained variance ratio: PC1={explained[0]:.4f}, PC2={explained[1]:.4f}")


def print_comparison_summary(
    rel_results: Dict[str, object],
    analogy_results: Dict[str, object],
    similarity_results: Dict[str, float | None],
):
    old_rel_cov = BASELINE_RESULTS["relatedness_coverage"]
    new_rel_cov = (rel_results["coverage"], rel_results["total"])

    old_ana_cov = BASELINE_RESULTS["analogy_coverage"]
    new_ana_cov = (analogy_results["coverage"], analogy_results["total"])

    old_acc = BASELINE_RESULTS["analogy_top5_accuracy"]
    new_acc = analogy_results["accuracy_top5"]

    print("\n=== OLD vs NEW Comparison ===")
    print(f"OLD window: {BASELINE_RESULTS['window']}")
    print(f"NEW window: {WINDOW_SIZE}")
    print(f"Relatedness coverage OLD: {old_rel_cov[0]}/{old_rel_cov[1]}")
    print(f"Relatedness coverage NEW: {new_rel_cov[0]}/{new_rel_cov[1]}")
    print(f"Analogy coverage OLD: {old_ana_cov[0]}/{old_ana_cov[1]}")
    print(f"Analogy coverage NEW: {new_ana_cov[0]}/{new_ana_cov[1]}")
    print(f"Top-5 analogy accuracy OLD: {old_acc}")
    print(f"Top-5 analogy accuracy NEW: {new_acc}")
    print(f"Top-5 analogy accuracy delta: {new_acc - old_acc:+.4f}")

    print("\nDirect similarity comparison:")
    for pair, old_val in BASELINE_RESULTS["direct_similarity"].items():
        new_val = similarity_results.get(pair)
        old_str = "OOV" if old_val is None else f"{old_val:.4f}"
        new_str = "OOV" if new_val is None else f"{new_val:.4f}"
        if old_val is None or new_val is None:
            delta_str = "n/a"
        else:
            delta_str = f"{(new_val - old_val):+.4f}"
        print(f"{pair:20s} OLD={old_str:>7s} NEW={new_str:>7s} DELTA={delta_str}")


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    ensure_nltk()

    print(f"Running SGNS experiment with window={WINDOW_SIZE}")
    print("Downloading Wikipedia article...")
    raw_text = fetch_wikipedia_article(WIKI_URL)

    print("Preprocessing text...")
    sentences = preprocess_text(raw_text)
    stats = corpus_stats(sentences)

    print("\n=== Corpus Stats ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\nTraining Skip-gram with Negative Sampling...")
    model = train_sgns(sentences)

    print("\nVocabulary size learned:", len(model.wv))

    probe_words = [
        "ai",
        "winter",
        "research",
        "funding",
        "expert",
        "systems",
        "neural",
        "network",
        "machine",
        "learning",
    ]
    print_top_neighbors(model, probe_words, topn=8)

    relatedness_test = [
        ("ai", "artificial", 0.95),
        ("ai", "intelligence", 0.95),
        ("expert", "systems", 0.92),
        ("machine", "learning", 0.94),
        ("neural", "network", 0.93),
        ("research", "funding", 0.70),
        ("winter", "funding", 0.60),
        ("symbolic", "reasoning", 0.78),
        ("ai", "kitchen", 0.05),
        ("neural", "tractor", 0.03),
        ("expert", "weather", 0.20),
        ("research", "hype", 0.40),
    ]

    rel_results = evaluate_relatedness(model, relatedness_test)

    print("\n=== Relatedness Test Set ===")
    print(f"Coverage: {rel_results['coverage']}/{rel_results['total']}")
    for w1, w2, gold, pred in rel_results["covered_items"]:
        print(f"{w1:10s} - {w2:10s} | gold={gold:.2f} pred={pred:.4f}")

    analogy_test = [
        ("machine", "learning", "neural", "network"),
        ("expert", "systems", "neural", "networks"),
        ("symbolic", "reasoning", "machine", "learning"),
        ("funding", "research", "hype", "expectations"),
    ]

    analogy_results = evaluate_analogies(model, analogy_test)

    print("\n=== Analogy Test Set ===")
    print(f"Coverage: {analogy_results['coverage']}/{analogy_results['total']}")
    print(f"Top-5 accuracy: {analogy_results['accuracy_top5']}")
    for item in analogy_results["details"]:
        print(json.dumps(item, ensure_ascii=False))

    print("\n=== Direct Similarity Checks ===")
    check_pairs = [
        ("ai", "intelligence"),
        ("machine", "learning"),
        ("expert", "systems"),
        ("ai", "kitchen"),
    ]
    similarity_results = {}
    for w1, w2 in check_pairs:
        pair_key = f"{w1}-{w2}"
        if has_word(model, w1) and has_word(model, w2):
            score = cosine(model, w1, w2)
            similarity_results[pair_key] = score
            print(f"{w1:10s} <-> {w2:10s}: {score:.4f}")
        else:
            similarity_results[pair_key] = None
            print(f"{w1:10s} <-> {w2:10s}: OOV")

    print_comparison_summary(rel_results, analogy_results, similarity_results)

    pca_words = select_pca_words(model, word_count=PCA_WORD_COUNT)
    print(f"\nPCA words used ({len(pca_words)}): {', '.join(pca_words)}")
    plot_pca_embeddings(model, pca_words, PCA_OUTPUT_PATH)

    model.save(OUTPUT_MODEL_PATH)
    print(f"\nSaved model to: {OUTPUT_MODEL_PATH}")
    print("\nDone.")


main()