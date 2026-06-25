import argparse
import json
import csv
import pandas as pd
from pathlib import Path

# Importing our implemented modules
from .preprocessing import clean_and_tokenize, generate_word_shingles
from .minhash import MinHashGenerator, calculate_minhash_similarity
from .lsh import LSHIndex

def read_text(path: str) -> str:
    """Reads a text file and returns its content."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def get_signature(text: str, generator: MinHashGenerator):
    """Helper: Converts raw text directly into a MinHash signature."""
    tokens = clean_and_tokenize(str(text))
    shingles = generate_word_shingles(tokens, shingle_size=3)
    return generator.generate_signature(shingles)

def handle_compare(args):
    """Command 1: Compare two specific files."""
    gen = MinHashGenerator()
    
    sig_a = get_signature(read_text(args.file_a), gen)
    sig_b = get_signature(read_text(args.file_b), gen)
    
    sim = calculate_minhash_similarity(sig_a, sig_b)
    
    # Save output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({"file_a": args.file_a, "file_b": args.file_b, "similarity": sim}, f, indent=4)
        
    print(f"[+] Similarity: {sim:.4f} | Saved to: {args.output}")

def handle_corpus(args):
    """Command 2: Find similar documents in a directory using LSH."""
    gen = MinHashGenerator()
    lsh = LSHIndex()
    db = {}
    
    # 1. Process all text files and insert into LSH
    for filepath in Path(args.data).glob("**/*.txt"):
        doc_id = filepath.name
        sig = get_signature(read_text(filepath), gen)
        db[doc_id] = sig
        lsh.insert(doc_id, sig)
        
    # 2. Get candidates and calculate exact similarity
    results = []
    for doc_a, doc_b in lsh.get_candidate_pairs():
        sim = calculate_minhash_similarity(db[doc_a], db[doc_b])
        if sim >= args.threshold:
            results.append((doc_a, doc_b, sim))
            
    # Save output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Doc_A", "Doc_B", "Similarity"])
        writer.writerows(sorted(results, key=lambda x: x[2], reverse=True))
        
    print(f"[+] Found {len(results)} pairs | Saved to: {args.output}")

def handle_pairs(args):
    """Command 3: Evaluate on labeled dataset (CSV)."""
    df = pd.read_csv(args.pairs).head(args.limit)
    gen = MinHashGenerator()
    
    tp = fp = fn = 0
    threshold = 0.3
    
    # Calculate TP, FP, FN
    for _, row in df.iterrows():
        sig_a = get_signature(row[args.text_col_a], gen)
        sig_b = get_signature(row[args.text_col_b], gen)
        sim = calculate_minhash_similarity(sig_a, sig_b)
        
        predicted = 1 if sim >= threshold else 0
        actual = int(row[args.label_col])
        
        if predicted == 1 and actual == 1: tp += 1
        elif predicted == 1 and actual == 0: fp += 1
        elif predicted == 0 and actual == 1: fn += 1

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    
    # Save output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows([
            ["Metric", "Value"], ["Precision", round(precision, 4)], 
            ["Recall", round(recall, 4)], ["F1_Score", round(f1, 4)]
        ])
        
    print(f"[+] Evaluation Done | F1-Score: {f1:.4f} | Saved to: {args.output}")

def main():
    parser = argparse.ArgumentParser(description="Plagiarism Detection CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Compare Command
    c1 = subparsers.add_parser("compare")
    c1.add_argument("--file-a", required=True)
    c1.add_argument("--file-b", required=True)
    c1.add_argument("--output", required=True)
    c1.set_defaults(func=handle_compare)

    # Corpus Command
    c2 = subparsers.add_parser("corpus")
    c2.add_argument("--data", required=True)
    c2.add_argument("--threshold", type=float, default=0.25)
    c2.add_argument("--output", required=True)
    c2.set_defaults(func=handle_corpus)

    # Pairs Command
    c3 = subparsers.add_parser("pairs")
    c3.add_argument("--pairs", required=True)
    c3.add_argument("--text-col-a", required=True)
    c3.add_argument("--text-col-b", required=True)
    c3.add_argument("--label-col", required=True)
    c3.add_argument("--limit", type=int, default=5000)
    c3.add_argument("--output", required=True)
    c3.set_defaults(func=handle_pairs)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()