import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("--- TEST FILE IS STARTING ---") 

from src.plagiarism_engine.preprocessing import clean_and_tokenize
print("\n--- Testing Preprocessing ---")
text = "Data )))Mining @is ‌‌#AMAZING٬!"
processed = clean_and_tokenize(text) 
print(f"Preprocessed tokens: {processed}")
print("Tokenize tests passed successfully!")



from src.plagiarism_engine.similarity import calculate_jaccard_similarity
print("\n--- Testing JaccardSimilarity ---")
set_a = {"data", "mining", "fun"}
set_b = {"data", "mining", "cool"}
sim = calculate_jaccard_similarity(set_a, set_b)
print(f"Similarity: {sim}")
assert sim > 0, "Similarity should be greater than 0"
print("Similarity tests passed successfully!")



from src.plagiarism_engine.minhash import MinHashGenerator, calculate_minhash_similarity
print("\n--- Testing MinHash ---")
shingles_a = {"data", "mining", "is", "awesome"}
shingles_b = {"data", "mining", "is", "awesome"} 
shingles_c = {"machine", "learning", "is", "hard"}
minhash = MinHashGenerator(num_permutations = 128)
sig_a = minhash.generate_signature(shingles_a)
sig_b = minhash.generate_signature(shingles_b)
sig_c = minhash.generate_signature(shingles_c)
sim_ab = calculate_minhash_similarity(sig_a, sig_b)
sim_ac = calculate_minhash_similarity(sig_a, sig_c)
print(f"Length of Signature A: {len(sig_a)}")
print(f"MinHash Similarity (A & B - Expected 1.0): {sim_ab}")
print(f"MinHash Similarity (A & C - Expected low): {sim_ac}")
assert len(sig_a) == 128, "Signature length should be 128"
assert sim_ab == 1.0, "Identical sets should have 1.0 similarity"
assert sim_ac < 1.0, "Different sets should have lower similarity"

print("MinHash tests passed successfully!")