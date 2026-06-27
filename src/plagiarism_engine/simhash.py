import hashlib
import math
from collections import Counter
from typing import Dict, List, Optional


class SimHashGenerator:
    """
    TF-IDF weighted SimHash implementation.
    Produces deterministic 64-bit fingerprints.
    """

    HASH_BITS = 64

    @staticmethod
    def _hash_token(token: str) -> int:
        """
        Stable 64-bit hash using SHA-256.

        We only use the first 8 bytes (64 bits) of the digest.
        """
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    @staticmethod
    def compute_idf(all_token_lists: List[List[str]]) -> Dict[str, float]:
        """
        Compute smoothed IDF values.

        IDF(t) = log((N + 1)/(df + 1)) + 1
        """
        N = len(all_token_lists)

        document_frequency = Counter()

        for tokens in all_token_lists:
            document_frequency.update(set(tokens))

        return {
            token: math.log((N + 1) / (df + 1)) + 1.0
            for token, df in document_frequency.items()
        }

    def generate_simhash(
        self,
        tokens: List[str],
        idf: Optional[Dict[str, float]] = None,
    ) -> int:
        """
        Generate a 64-bit SimHash fingerprint.
        """

        if not tokens:
            return 0

        tf = Counter(tokens)

        vector = [0.0] * self.HASH_BITS

        for token, freq in tf.items():

            # logarithmic TF
            tf_weight = 1.0 + math.log(freq)

            idf_weight = idf.get(token, 1.0) if idf else 1.0

            weight = tf_weight * idf_weight

            h = self._hash_token(token)

            for bit in range(self.HASH_BITS):

                if (h >> bit) & 1:
                    vector[bit] += weight
                else:
                    vector[bit] -= weight

        fingerprint = 0

        for bit in range(self.HASH_BITS):
            if vector[bit] > 0:
                fingerprint |= 1 << bit

        return fingerprint


def hamming_distance(h1: int, h2: int) -> int:
    """
    Number of differing bits.
    """
    return (h1 ^ h2).bit_count()


def simhash_similarity(h1: int, h2: int) -> float:
    """
    Similarity score in [0,1].
    """

    if h1 == 0 and h2 == 0:
        return 1.0

    distance = hamming_distance(h1, h2)

    return 1.0 - (distance / SimHashGenerator.HASH_BITS)