"""
Exact line deduplication and MinHash + LSH fuzzy document deduplication.
"""

import hashlib
import re
from typing import Callable, Iterable


def exact_line_deduplication(corpus: list[str]) -> list[str]:
    """
    Performs exact line-level deduplication across a corpus of documents.
    
    Removes lines that appear more than once across documents (e.g. repeated
    navbars, headers, footers, copyright notices).
    
    Args:
        corpus: List of document string texts.
        
    Returns:
        List of deduplicated document texts.
    """
    # Count line frequencies across all documents
    line_counts: dict[str, int] = {}
    for doc in corpus:
        for line in doc.splitlines():
            cleaned = line.strip()
            if cleaned:
                line_counts[cleaned] = line_counts.get(cleaned, 0) + 1

    # Filter out lines that occur more than once globally
    deduped_docs = []
    for doc in corpus:
        kept_lines = []
        for line in doc.splitlines():
            cleaned = line.strip()
            if not cleaned or line_counts.get(cleaned, 0) == 1:
                kept_lines.append(line)
        deduped_docs.append("\n".join(kept_lines))

    return deduped_docs


class UnionFind:
    """Disjoint-set / Union-Find data structure for graph component detection."""

    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


class MinHashLSH:
    """
    MinHash and Locality-Sensitive Hashing (LSH) for fuzzy document deduplication.
    """

    PRIME_31 = 2147483647  # Mersenne Prime 2^31 - 1

    def __init__(
        self,
        num_hashes: int = 64,
        num_bands: int = 16,
        shingle_size: int = 5,
        seed: int = 42,
    ):
        """
        Args:
            num_hashes: Total number of MinHash hash functions (K).
            num_bands: Number of LSH bands (B). Must divide num_hashes.
            shingle_size: Length k of word shingles.
            seed: Random seed for hash parameters a_i, b_i.
        """
        assert num_hashes % num_bands == 0, "num_hashes must be divisible by num_bands"
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.shingle_size = shingle_size

        # Generate deterministic linear hash function parameters: (a*x + b) % PRIME
        import random
        rng = random.Random(seed)
        self.a_coefficients = [rng.randint(1, self.PRIME_31 - 1) for _ in range(num_hashes)]
        self.b_coefficients = [rng.randint(0, self.PRIME_31 - 1) for _ in range(num_hashes)]

    def get_shingles(self, text: str) -> set[str]:
        """Extract word k-shingles from text."""
        # Normalize text and extract words
        words = re.findall(r"\w+", text.lower())
        if len(words) < self.shingle_size:
            return set([" ".join(words)]) if words else set()

        shingles = set()
        for i in range(len(words) - self.shingle_size + 1):
            shingle = " ".join(words[i : i + self.shingle_size])
            shingles.add(shingle)
        return shingles

    def shingle_hash(self, shingle: str) -> int:
        """Hash shingle string to a 32-bit positive integer."""
        md5 = hashlib.md5(shingle.encode("utf-8")).hexdigest()
        return int(md5[:8], 16) % self.PRIME_31

    def compute_minhash_signature(self, shingles: set[str]) -> list[int]:
        """Compute MinHash signature vector for a shingle set."""
        if not shingles:
            return [0] * self.num_hashes

        shingle_hashes = [self.shingle_hash(s) for s in shingles]

        signature = []
        for i in range(self.num_hashes):
            a = self.a_coefficients[i]
            b = self.b_coefficients[i]
            # Find minimum hash value across all shingles for this hash function
            min_val = min((a * h + b) % self.PRIME_31 for h in shingle_hashes)
            signature.append(min_val)

        return signature

    @staticmethod
    def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
        """Compute exact Jaccard similarity between two sets."""
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def deduplicate_dataset(
        self, documents: list[str], similarity_threshold: float = 0.7
    ) -> list[str]:
        """
        Deduplicate a list of documents using MinHash + LSH and graph connected components.
        
        Args:
            documents: List of input document strings.
            similarity_threshold: Minimum Jaccard similarity threshold to consider near-duplicate.
            
        Returns:
            Deduplicated list of document strings.
        """
        if not documents:
            return []

        # 1. Compute shingles and MinHash signatures for all documents
        shingle_sets = [self.get_shingles(doc) for doc in documents]
        signatures = [self.compute_minhash_signature(s_set) for s_set in shingle_sets]

        # 2. Map signatures to LSH buckets per band
        # bucket_key -> list of document indices
        buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}

        for doc_idx, sig in enumerate(signatures):
            for band_idx in range(self.num_bands):
                start = band_idx * self.rows_per_band
                end = start + self.rows_per_band
                band_tuple = tuple(sig[start:end])
                bucket_key = (band_idx, band_tuple)

                if bucket_key not in buckets:
                    buckets[bucket_key] = []
                buckets[bucket_key].append(doc_idx)

        # 3. Find candidate pairs sharing at least one LSH bucket
        candidate_pairs: set[tuple[int, int]] = set()
        for doc_indices in buckets.values():
            if len(doc_indices) > 1:
                for i in range(len(doc_indices)):
                    for j in range(i + 1, len(doc_indices)):
                        doc_a, doc_b = doc_indices[i], doc_indices[j]
                        if doc_a > doc_b:
                            doc_a, doc_b = doc_b, doc_a
                        candidate_pairs.add((doc_a, doc_b))

        # 4. Verify candidate pairs using exact Jaccard similarity and build graph
        uf = UnionFind(len(documents))
        for doc_a, doc_b in candidate_pairs:
            sim = self.jaccard_similarity(shingle_sets[doc_a], shingle_sets[doc_b])
            if sim >= similarity_threshold:
                uf.union(doc_a, doc_b)

        # 5. Retain one representative document per connected component
        seen_roots: set[int] = set()
        deduplicated_docs = []

        for doc_idx in range(len(documents)):
            root = uf.find(doc_idx)
            if root not in seen_roots:
                seen_roots.add(root)
                deduplicated_docs.append(documents[doc_idx])

        return deduplicated_docs
