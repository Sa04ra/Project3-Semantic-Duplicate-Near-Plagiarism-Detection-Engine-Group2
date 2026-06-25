from collections import defaultdict
from typing import List, Set, Tuple, Dict
import itertools

class LSHIndex:
    """
    Locality Sensitive Hashing (LSH) implementation for MinHash signatures.
    Divides signatures into bands and hashes them into buckets to find candidate pairs.
    """
    
    def __init__(self, signature_length: int = 128, num_bands: int = 32):
        """
        Initializes the LSH index.
        
        Args:
            signature_length: The length of the MinHash signature (default 128).
            num_bands: The number of bands to divide the signature into.
                       Note: signature_length must be perfectly divisible by num_bands.
        """
        if signature_length % num_bands != 0:
            raise ValueError(f"signature_length ({signature_length}) must be divisible by num_bands ({num_bands})")
            
        self.signature_length = signature_length
        self.num_bands = num_bands
        self.rows_per_band = signature_length // num_bands
        
        # A list of dictionaries to represent the buckets for each band.
        # Index of the list corresponds to the band number.
        # Key in the dictionary: a tuple representing the sub-vector (hash of the band)
        # Value in the dictionary: a set of document IDs that fall into this bucket
        self.bands = [defaultdict(set) for _ in range(self.num_bands)]

    def insert(self, doc_id: str, signature: List[int]) -> None:
        """
        Inserts a document's MinHash signature into the LSH buckets.
        """
        if len(signature) != self.signature_length:
            raise ValueError(f"Expected signature of length {self.signature_length}, got {len(signature)}")
            
        # Divide the signature into bands
        for band_idx in range(self.num_bands):
            start_idx = band_idx * self.rows_per_band
            end_idx = start_idx + self.rows_per_band
            
            # Extract the specific band from the signature
            # We convert the list slice to a tuple so it can be used as a dictionary key (hashable)
            band_tuple = tuple(signature[start_idx:end_idx])
            
            # Assign the document ID to the corresponding bucket in this band
            self.bands[band_idx][band_tuple].add(doc_id)

    def query(self, signature: List[int]) -> Set[str]:
        """
        Queries the LSH index for a given signature and returns candidate document IDs
        that share at least one bucket with the input signature.
        """
        candidates = set()
        
        for band_idx in range(self.num_bands):
            start_idx = band_idx * self.rows_per_band
            end_idx = start_idx + self.rows_per_band
            
            band_tuple = tuple(signature[start_idx:end_idx])
            
            # If this band tuple exists in the current band's buckets,
            # add all documents inside that bucket to our candidates list
            if band_tuple in self.bands[band_idx]:
                candidates.update(self.bands[band_idx][band_tuple])
                
        return candidates

    def get_candidate_pairs(self) -> Set[Tuple[str, str]]:
        """
        Scans all buckets in all bands and returns every pair of documents
        that fell into the same bucket at least once.
        This drastically reduces the comparisons compared to "all-to-all".
        """
        candidate_pairs = set()
        
        for band_buckets in self.bands:
            for bucket_docs in band_buckets.values():
                # If a bucket contains more than one document, they are candidates
                if len(bucket_docs) > 1:
                    # Sort documents to ensure pair (A, B) is identical to (B, A)
                    sorted_docs = sorted(list(bucket_docs))
                    
                    # Generate all unique combinations of size 2 from the documents in this bucket
                    for pair in itertools.combinations(sorted_docs, 2):
                        candidate_pairs.add(pair)
                        
        return candidate_pairs