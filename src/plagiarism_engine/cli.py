"""
cli.py — Command-line interface for the Plagiarism Detection Engine.

Three commands:
  compare   Compare two text files and show exact + approximate similarity.
  corpus    Index a directory of .txt files and find similar pairs via LSH.
  pairs     Evaluate one algorithm on a labeled CSV dataset and report metrics.

Usage examples:
  python -m plagiarism_engine.cli compare \
      --file-a data/sample_corpus/doc_01.txt \
      --file-b data/sample_corpus/doc_02.txt \
      --output outputs/compare.json

  python -m plagiarism_engine.cli corpus \
      --data data/sample_corpus \
      --threshold 0.25 \
      --shingle-size 3 \
      --output outputs/candidates.csv

  python -m plagiarism_engine.cli pairs \
      --pairs data/raw/quora/train.csv \
      --text-col-a question1 --text-col-b question2 \
      --label-col is_duplicate \
      --algorithm minhash \
      --threshold 0.30 \
      --limit 5000 \
      --output outputs/metrics.csv
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

from .minhash import MinHashGenerator, calculate_minhash_similarity
from .lsh import LSHIndex
from .preprocessing import clean_and_tokenize, generate_word_shingles
from .similarity import calculate_jaccard_similarity
from .simhash import SimHashGenerator, simhash_similarity


# ──────────────────────────────────────────────────────────────────
# Token cache
# Tokenisation is the slowest step. Caching each unique string means
# it is tokenized exactly once per run — 2-3x speedup on large datasets.
# ──────────────────────────────────────────────────────────────────

_token_cache: Dict[str, List[str]] = {}


def _get_tokens(text: str) -> List[str]:
    """Return cached token list for *text*, tokenizing on first access."""
    key = str(text)
    if key not in _token_cache:
        _token_cache[key] = clean_and_tokenize(key)
    return _token_cache[key]


# ──────────────────────────────────────────────────────────────────
# File I/O helpers
# ──────────────────────────────────────────────────────────────────

def _read_text(path: str) -> str:
    """
    Read a UTF-8 text file.
    Raises IOError with a human-readable message on any failure.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise IOError(f"File not found: {path}")
    except Exception as e:
        raise IOError(f"Could not read {path}: {e}")


def _save_json(data: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def _save_csv(rows: List[list], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


# ──────────────────────────────────────────────────────────────────
# Command: compare
# ──────────────────────────────────────────────────────────────────

def handle_compare(args: argparse.Namespace) -> None:
    """
    Compare two documents and report similarity scores.

    --algorithm minhash  ->  exact Jaccard  +  MinHash approximation
    --algorithm simhash  ->  exact Jaccard  +  SimHash / Hamming similarity

    The exact Jaccard is always included as a ground-truth reference.
    For SimHash, IDF is computed from the two documents themselves so
    that rare shared tokens receive higher weight.
    """
    _token_cache.clear()

    try:
        text_a = _read_text(args.file_a)
        text_b = _read_text(args.file_b)
    except IOError as e:
        print(f"[!] {e}", file=sys.stderr)
        return

    tokens_a = _get_tokens(text_a)
    tokens_b = _get_tokens(text_b)
    shingles_a: Set[str] = generate_word_shingles(tokens_a, args.shingle_size)
    shingles_b: Set[str] = generate_word_shingles(tokens_b, args.shingle_size)

    # Exact Jaccard is always computed as ground truth
    exact_jaccard = calculate_jaccard_similarity(shingles_a, shingles_b)

    result: dict = {
        "file_a": args.file_a,
        "file_b": args.file_b,
        "algorithm": args.algorithm,
        "parameters": {
            "shingle_size": args.shingle_size,
            "signature_length": 128,
            "simhash_bits": 64,
        },
        "exact_jaccard": round(exact_jaccard, 4),
    }

    if args.algorithm == "minhash":
        gen = MinHashGenerator()
        sig_a = gen.generate_signature(shingles_a)
        sig_b = gen.generate_signature(shingles_b)
        mh_sim = calculate_minhash_similarity(sig_a, sig_b)
        result["minhash_similarity"] = round(mh_sim, 4)
        result["approximation_error"] = round(abs(exact_jaccard - mh_sim), 4)
        print(f"[+] Exact Jaccard       : {exact_jaccard:.4f}")
        print(f"[+] MinHash similarity  : {mh_sim:.4f}")
        print(f"[+] Approximation error : {result['approximation_error']:.4f}")

    else:  # simhash
        gen = SimHashGenerator()
        # IDF from these two docs so shared-rare tokens get higher weight
        idf = gen.compute_idf([tokens_a, tokens_b])
        fp_a = gen.generate_simhash(tokens_a, idf)
        fp_b = gen.generate_simhash(tokens_b, idf)
        sh_sim = simhash_similarity(fp_a, fp_b)
        result["simhash_similarity"] = round(sh_sim, 4)
        print(f"[+] Exact Jaccard       : {exact_jaccard:.4f}")
        print(f"[+] SimHash similarity  : {sh_sim:.4f}")

    _save_json(result, args.output)
    print(f"[+] Saved to            : {args.output}")


# ──────────────────────────────────────────────────────────────────
# Command: corpus
# ──────────────────────────────────────────────────────────────────

def handle_corpus(args: argparse.Namespace) -> None:
    """
    Index all .txt files in a directory using MinHash + LSH and output
    every candidate pair whose MinHash similarity meets the threshold.

    LSH is configured with 64 bands (2 rows/band) so its S-curve
    threshold sits near 0.25, matching the CLI default.

    doc_id is the path relative to --data, not just the filename, so
    files with the same name in different subdirectories never collide.
    """
    _token_cache.clear()

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"[!] Directory not found: {args.data}", file=sys.stderr)
        return

    txt_files = list(data_dir.glob("**/*.txt"))
    if not txt_files:
        print(f"[!] No .txt files found in {args.data}", file=sys.stderr)
        return

    mh_gen = MinHashGenerator()
    # 64 bands x 2 rows/band -> LSH threshold ~0.25 (matches CLI default)
    lsh = LSHIndex(signature_length=128, num_bands=64)
    db: Dict[str, List[int]] = {}

    print(f"[*] Indexing {len(txt_files)} documents...")
    skipped = 0
    for filepath in txt_files:
        doc_id = str(filepath.relative_to(data_dir))  # no filename collisions
        try:
            text = _read_text(str(filepath))
        except IOError as e:
            print(f"    [!] Skipping {doc_id}: {e}", file=sys.stderr)
            skipped += 1
            continue
        tokens = _get_tokens(text)
        shingles = generate_word_shingles(tokens, args.shingle_size)
        sig = mh_gen.generate_signature(shingles)
        db[doc_id] = sig
        lsh.insert(doc_id, sig)

    indexed = len(db)
    print(f"[+] Indexed {indexed} documents" +
          (f" ({skipped} skipped due to read errors)" if skipped else ""))

    results = []
    for doc_a, doc_b in lsh.get_candidate_pairs():
        sim = calculate_minhash_similarity(db[doc_a], db[doc_b])
        if sim >= args.threshold:
            results.append([doc_a, doc_b, round(sim, 4)])

    results.sort(key=lambda x: x[2], reverse=True)
    _save_csv([["Doc_A", "Doc_B", "MinHash_Similarity"]] + results, args.output)
    print(f"[+] Found {len(results)} pairs above threshold {args.threshold}")
    print(f"[+] Saved to: {args.output}")


# ──────────────────────────────────────────────────────────────────
# Command: pairs — internal helpers
# ──────────────────────────────────────────────────────────────────

def _compute_metrics(preds: List[int], labels: List[int]) -> tuple:
    """Return (precision, recall, f1) from binary predictions and labels."""
    tp = fp = fn = 0
    for p, a in zip(preds, labels):
        if   p == 1 and a == 1: tp += 1
        elif p == 1 and a == 0: fp += 1
        elif p == 0 and a == 1: fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return round(prec, 4), round(rec, 4), round(f1, 4)


# ──────────────────────────────────────────────────────────────────
# Command: pairs
# ──────────────────────────────────────────────────────────────────

def handle_pairs(args: argparse.Namespace) -> None:
    """
    Evaluate one algorithm on a labeled CSV of text pairs.

    Run the command twice — once with --algorithm minhash and once
    with --algorithm simhash — to get the side-by-side comparison
    required for the technical report.

    Output CSV columns:
      Method, Threshold, Precision, Recall, F1_Score,
      Time_sec, Pairs_per_sec, Pairs_evaluated
    """
    _token_cache.clear()

    try:
        df = pd.read_csv(args.pairs).head(args.limit)
    except FileNotFoundError:
        print(f"[!] Dataset file not found: {args.pairs}", file=sys.stderr)
        return
    except Exception as e:
        print(f"[!] Could not load dataset: {e}", file=sys.stderr)
        return

    n = len(df)
    print(f"[*] Loaded {n} pairs")

    # Pre-tokenize every text exactly once using the cache
    all_token_lists: List[List[str]] = []
    for _, row in df.iterrows():
        all_token_lists.append(_get_tokens(str(row[args.text_col_a])))
        all_token_lists.append(_get_tokens(str(row[args.text_col_b])))

    labels = [int(row[args.label_col]) for _, row in df.iterrows()]
    preds: List[int] = []

    if args.algorithm == "minhash":
        print(f"[*] Running MinHash (threshold={args.threshold})...")
        gen = MinHashGenerator()
        t0 = time.perf_counter()
        for _, row in df.iterrows():
            shingles_a = generate_word_shingles(
                _get_tokens(str(row[args.text_col_a])), args.shingle_size)
            shingles_b = generate_word_shingles(
                _get_tokens(str(row[args.text_col_b])), args.shingle_size)
            sig_a = gen.generate_signature(shingles_a)
            sig_b = gen.generate_signature(shingles_b)
            sim = calculate_minhash_similarity(sig_a, sig_b)
            preds.append(1 if sim >= args.threshold else 0)
        t_elapsed = time.perf_counter() - t0
        method_label = "MinHash"

    else:  # simhash
        print(f"[*] Building IDF table from {len(all_token_lists)} token lists...")
        gen_sh = SimHashGenerator()
        idf = gen_sh.compute_idf(all_token_lists)
        print(f"[*] Running SimHash (threshold={args.threshold})...")
        t0 = time.perf_counter()
        for _, row in df.iterrows():
            fp_a = gen_sh.generate_simhash(
                _get_tokens(str(row[args.text_col_a])), idf)
            fp_b = gen_sh.generate_simhash(
                _get_tokens(str(row[args.text_col_b])), idf)
            sim = simhash_similarity(fp_a, fp_b)
            preds.append(1 if sim >= args.threshold else 0)
        t_elapsed = time.perf_counter() - t0
        method_label = "SimHash"

    prec, rec, f1 = _compute_metrics(preds, labels)
    pairs_per_sec = round(n / t_elapsed, 1) if t_elapsed > 0 else 0.0

    _save_csv(
        [
            ["Method", "Threshold", "Precision", "Recall", "F1_Score",
             "Time_sec", "Pairs_per_sec", "Pairs_evaluated"],
            [method_label, args.threshold, prec, rec, f1,
             round(t_elapsed, 3), pairs_per_sec, n],
        ],
        args.output,
    )

    print(f"\n[+] {method_label} results:")
    print(f"    Precision      : {prec}")
    print(f"    Recall         : {rec}")
    print(f"    F1 Score       : {f1}")
    print(f"    Time           : {t_elapsed:.2f}s  ({pairs_per_sec} pairs/sec)")
    print(f"[+] Saved to: {args.output}")


# ──────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="plagiarism_engine",
        description="Semantic Duplicate & Near-Plagiarism Detection Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── compare ───────────────────────────────────────────────────
    c1 = subparsers.add_parser(
        "compare",
        help="Compare two text files (exact Jaccard + chosen algorithm)",
    )
    c1.add_argument("--file-a",       required=True,
                    help="Path to first document")
    c1.add_argument("--file-b",       required=True,
                    help="Path to second document")
    c1.add_argument("--algorithm",    choices=["minhash", "simhash"],
                    default="minhash",
                    help="Algorithm to use (default: minhash)")
    c1.add_argument("--shingle-size", type=int, default=3,
                    help="Word shingle size (default: 3)")
    c1.add_argument("--output",       required=True,
                    help="Output JSON path")
    c1.set_defaults(func=handle_compare)

    # ── corpus ────────────────────────────────────────────────────
    c2 = subparsers.add_parser(
        "corpus",
        help="Index a directory of .txt files and find similar pairs via LSH",
    )
    c2.add_argument("--data",         required=True,
                    help="Directory containing .txt files")
    c2.add_argument("--threshold",    type=float, default=0.25,
                    help="Minimum similarity to report (default: 0.25)")
    c2.add_argument("--shingle-size", type=int,   default=3,
                    help="Word shingle size (default: 3)")
    c2.add_argument("--output",       required=True,
                    help="Output CSV path")
    c2.set_defaults(func=handle_corpus)

    # ── pairs ─────────────────────────────────────────────────────
    c3 = subparsers.add_parser(
        "pairs",
        help="Evaluate MinHash or SimHash on a labeled CSV dataset",
    )
    c3.add_argument("--pairs",        required=True,
                    help="Path to labeled CSV file")
    c3.add_argument("--text-col-a",   required=True,
                    help="Column name for text A")
    c3.add_argument("--text-col-b",   required=True,
                    help="Column name for text B")
    c3.add_argument("--label-col",    required=True,
                    help="Column name for duplicate label (0 or 1)")
    c3.add_argument("--algorithm",    choices=["minhash", "simhash"],
                    default="minhash",
                    help="Algorithm to evaluate (default: minhash)")
    c3.add_argument("--threshold",    type=float, default=0.30,
                    help="Similarity threshold (default: 0.30 for MinHash, "
                         "try 0.85 for SimHash)")
    c3.add_argument("--shingle-size", type=int,   default=3,
                    help="Word shingle size for MinHash (default: 3)")
    c3.add_argument("--limit",        type=int,   default=5000,
                    help="Max rows to evaluate (default: 5000)")
    c3.add_argument("--output",       required=True,
                    help="Output CSV path for metrics")
    c3.set_defaults(func=handle_pairs)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()