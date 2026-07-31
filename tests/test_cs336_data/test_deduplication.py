"""
Unit tests for exact line and MinHash LSH fuzzy deduplication.
"""

from cs336_data.deduplication import exact_line_deduplication, MinHashLSH


def test_exact_line_deduplication():
    doc1 = "Header Menu\nUnique content in document one.\nFooter Copyright 2026"
    doc2 = "Header Menu\nUnique content in document two.\nFooter Copyright 2026"
    
    deduped = exact_line_deduplication([doc1, doc2])
    assert len(deduped) == 2
    assert "Header Menu" not in deduped[0]
    assert "Footer Copyright 2026" not in deduped[0]
    assert "Unique content in document one." in deduped[0]
    assert "Unique content in document two." in deduped[1]


def test_minhash_signature_computation():
    minhash = MinHashLSH(num_hashes=32, num_bands=8, shingle_size=3)
    text1 = "large language model pretraining data processing pipeline"
    shingles1 = minhash.get_shingles(text1)
    sig1 = minhash.compute_minhash_signature(shingles1)
    
    assert len(sig1) == 32
    assert all(isinstance(x, int) for x in sig1)


def test_minhash_fuzzy_deduplication():
    minhash = MinHashLSH(num_hashes=32, num_bands=8, shingle_size=3)
    
    doc1 = "The quick brown fox jumps over the lazy dog in a sunny forest today."
    doc2 = "The quick brown fox jumps over the lazy dog in a sunny forest yesterday."
    doc3 = "Quantum computing relies on qubits and superposition states for computation."

    deduped = minhash.deduplicate_dataset([doc1, doc2, doc3], similarity_threshold=0.6)
    
    # doc1 and doc2 are near-duplicates, doc3 is distinct
    assert len(deduped) == 2
    assert doc3 in deduped
