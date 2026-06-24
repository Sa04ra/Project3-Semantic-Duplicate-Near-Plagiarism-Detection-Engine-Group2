import random
import binascii
from typing import List, Set

class MinHashGenerator:
    """
    MinHashGenerator implements the MinHash signature generation from scratch.
    It creates a family of hash functions of the form: h(x) = (a * x + b) % p
    where p is a large prime number, and a, b are random coefficients.
    """
    def __init__(self, num_permutations: int = 128, seed: int = 42):
        self.num_permutations = num_permutations
        
        # A large Mersenne prime number (2^31 - 1) for 32-bit hashing
        self.prime = 2147483647
        
        # Set the random seed to guarantee reproducible results across runs
        random.seed(seed)
        
        # Generate random coefficients for the hash function family
        # Coefficient 'a' must be a non-zero integer less than the prime
        self.a_coefficients = [random.randint(1, self.prime - 1) for _ in range(num_permutations)]
        self.b_coefficients = [random.randint(0, self.prime - 1) for _ in range(num_permutations)]

    def _hash_shingle(self, shingle: str) -> int:
        """
        Converts a string shingle into a stable 32-bit unsigned integer using CRC32.
        Python's built-in hash() is intentionally avoided because it is randomized 
        per session, which breaks the consistency required for CLI tools.
        """
        # Convert the string shingle into bytes
        bytes_data = shingle.encode('utf-8')
    
        # Compute a 32-bit CRC hash
        raw_hash = binascii.crc32(bytes_data)
    
        # Ensure the hash remains within the 32-bit unsigned integer range (0 to 2^32 - 1)
        # This prevents potential negative values from the CRC32 function
        final_hash = raw_hash & 0xffffffff
    
        return final_hash

    def generate_signature(self, shingles: Set[str]) -> List[int]:
        """
        Generates a MinHash signature vector for a given set of string shingles.
        Returns a list of integers with a length equal to num_permutations.
        """
        if not shingles:
            # Return a vector filled with the prime value as placeholder for empty inputs
            return [self.prime] * self.num_permutations

        # Initialize the signature vector with 'infinity' (the maximum possible prime value)
        signature = [self.prime] * self.num_permutations

        for shingle in shingles:
            # 1. Convert the string shingle to its base 32-bit integer hash
            shingle_hash = self._hash_shingle(shingle)
            
            # 2. Pass the base hash through the family of hash functions
            for i in range(self.num_permutations):
                # Linear universal hashing: h(x) = (a * x + b) % p
                permuted_hash = (self.a_coefficients[i] * shingle_hash + self.b_coefficients[i]) % self.prime
                
                # MinHash property: keep the minimum hash value seen so far
                if permuted_hash < signature[i]:
                    signature[i] = permuted_hash

        return signature

def calculate_minhash_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """
    Estimates the Jaccard similarity between two documents using their MinHash signatures.
    It calculates the ratio of matching positions between the two signature vectors.
    """
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0

    # Count how many slots in both signatures have the exact same value
    matching_slots = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    
    # Approximate Jaccard similarity = (number of matches) / (total signature length)
    return matching_slots / len(sig_a)