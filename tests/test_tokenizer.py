"""
Unit tests for BPETokenizer (Stanford CS336 Assignment 1).
"""

import os
import pytest
from cs336_basics.tokenizer import BPETokenizer


@pytest.fixture
def sample_text():
    return (
        "hug hug hug hug hug hug "
        "pug pug pug pug "
        "pun pun "
        "bun bun "
        "huge huge "
    )


def test_bpe_training_and_roundtrip(sample_text):
    # Train BPE with a small target vocabulary
    tokenizer = BPETokenizer.train(sample_text, vocab_size=300, special_tokens=["<|endoftext|>"])

    assert len(tokenizer.vocab) >= 256
    assert "<|endoftext|>" in tokenizer.special_tokens

    # Test encoding and decoding roundtrip
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    assert decoded == sample_text


def test_bpe_special_tokens():
    text = "Hello world! <|endoftext|> CS336 is awesome."
    tokenizer = BPETokenizer.train("Hello world! CS336 is awesome.", vocab_size=280, special_tokens=["<|endoftext|>"])

    # Test allowed_special="all"
    encoded = tokenizer.encode(text, allowed_special="all")
    assert tokenizer.special_tokens["<|endoftext|>"] in encoded

    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_bpe_save_and_load(tmp_path, sample_text):
    tokenizer = BPETokenizer.train(sample_text, vocab_size=280, special_tokens=["<|endoftext|>"])

    vocab_file = tmp_path / "vocab.json"
    merges_file = tmp_path / "merges.json"

    tokenizer.save(vocab_file, merges_file)
    assert os.path.exists(vocab_file)
    assert os.path.exists(merges_file)

    loaded_tokenizer = BPETokenizer.from_files(vocab_file, merges_file)
    
    test_str = "hug pug pun bun huge <|endoftext|>"
    orig_encoded = tokenizer.encode(test_str, allowed_special="all")
    loaded_encoded = loaded_tokenizer.encode(test_str, allowed_special="all")

    assert orig_encoded == loaded_encoded
    assert loaded_tokenizer.decode(loaded_encoded) == test_str


def test_bpe_utf8_multibyte_roundtrip():
    text = "Olá mundo! 🚀 CS336 LLM da Stanford. Em português: 'ação', 'coração'."
    tokenizer = BPETokenizer.train(text, vocab_size=300)

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text
