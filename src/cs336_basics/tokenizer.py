"""
Byte-Pair Encoding (BPE) Tokenizer implementation for Stanford CS336.
Supports GPT-2 regex pre-tokenization, training from raw text, encoding, decoding,
special token handling, and serialization/deserialization.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Set, Tuple, Union

import regex

# Default GPT-2 pre-tokenization regex split pattern
GPT2_SPLIT_REGEX = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    """
    Byte-Pair Encoding Tokenizer.
    """

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Dict[str, int] | None = None,
    ):
        """
        Initialize BPE Tokenizer with vocabulary mapping, merge rules, and special tokens.

        Args:
            vocab: Mapping from token_id -> byte sequence (bytes).
            merges: Ordered list of byte pairs merged during training.
            special_tokens: Mapping from special token string -> token_id.
        """
        self.vocab = dict(vocab)
        self.bytes_to_id: Dict[bytes, int] = {b: i for i, b in self.vocab.items()}
        self.merges = list(merges)
        
        # Rank of pair merges (lower index = higher priority)
        self.bpe_ranks: Dict[Tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }

        self.special_tokens: Dict[str, int] = special_tokens or {}
        self.inverse_special_tokens: Dict[int, str] = {
            idx: token for token, idx in self.special_tokens.items()
        }

        # Regex compiled for pre-tokenization split
        self.pat = regex.compile(GPT2_SPLIT_REGEX)

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        special_tokens: List[str] | None = None,
    ) -> BPETokenizer:
        """
        Train BPE Tokenizer on raw text.

        Args:
            text: Input corpus text.
            vocab_size: Target total vocabulary size (including byte tokens & merges, excluding special tokens).
            special_tokens: List of special token strings to add to tokenizer.

        Returns:
            Trained BPETokenizer instance.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 to cover all byte values.")

        # Base vocabulary: 256 byte tokens (0..255)
        vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        
        # Pre-tokenize corpus using GPT-2 regex
        words = regex.findall(GPT2_SPLIT_REGEX, text)

        # Count frequencies of word byte-sequences
        word_counts: Dict[Tuple[bytes, ...], int] = {}
        for word in words:
            byte_tuple = tuple(bytes([b]) for b in word.encode("utf-8"))
            word_counts[byte_tuple] = word_counts.get(byte_tuple, 0) + 1

        merges: List[Tuple[bytes, bytes]] = []

        num_merges = vocab_size - 256
        for step in range(num_merges):
            # Count pair frequencies across all words
            pair_counts: Dict[Tuple[bytes, bytes], int] = {}
            for word, freq in word_counts.items():
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq

            if not pair_counts:
                break

            # Find pair with max frequency (break ties by lexicographical order for determinism)
            best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
            if pair_counts[best_pair] == 0:
                break

            # Merge best_pair in all words
            merged_token = best_pair[0] + best_pair[1]
            new_id = len(vocab)
            vocab[new_id] = merged_token
            merges.append(best_pair)

            new_word_counts: Dict[Tuple[bytes, ...], int] = {}
            for word, freq in word_counts.items():
                new_word: List[bytes] = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                        new_word.append(merged_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_counts[tuple(new_word)] = freq
            word_counts = new_word_counts

        # Construct special tokens map starting at index after base vocab & merges
        special_token_map: Dict[str, int] = {}
        if special_tokens:
            current_id = max(vocab.keys()) + 1
            for token in special_tokens:
                special_token_map[token] = current_id
                current_id += 1

        return cls(vocab=vocab, merges=merges, special_tokens=special_token_map)

    def _encode_chunk(self, chunk_bytes: bytes) -> List[int]:
        """
        Encode a single pre-tokenized byte chunk using learned BPE ranks.
        """
        if not chunk_bytes:
            return []

        # Represent as list of individual byte objects
        parts: List[bytes] = [bytes([b]) for b in chunk_bytes]

        while len(parts) >= 2:
            # Find candidate pairs and their rank
            pairs = [(parts[i], parts[i + 1]) for i in range(len(parts) - 1)]
            
            # Find pair with lowest merge rank (highest priority)
            best_pair = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            
            if best_pair not in self.bpe_ranks:
                break

            # Merge all occurrences of best_pair in parts
            merged_token = best_pair[0] + best_pair[1]
            new_parts: List[bytes] = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best_pair:
                    new_parts.append(merged_token)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts

        return [self.bytes_to_id[p] for p in parts]

    def encode(
        self,
        text: str,
        allowed_special: Union[Set[str], str] = "none",
    ) -> List[int]:
        """
        Encode text into token IDs.

        Args:
            text: Input string.
            allowed_special: Set of special token strings permitted in text, or "all"/"none".

        Returns:
            List of integer token IDs.
        """
        if allowed_special == "all":
            special_set = set(self.special_tokens.keys())
        elif allowed_special == "none" or allowed_special is None:
            special_set = set()
        elif isinstance(allowed_special, set):
            special_set = allowed_special
        else:
            raise ValueError(f"Invalid allowed_special value: {allowed_special}")

        if not special_set:
            # Pre-tokenize and encode normal text
            ids: List[int] = []
            for chunk in self.pat.findall(text):
                chunk_bytes = chunk.encode("utf-8")
                ids.extend(self._encode_chunk(chunk_bytes))
            return ids

        # Split text on special tokens
        special_pattern = "|".join(re.escape(tok) for tok in sorted(special_set, key=len, reverse=True))
        special_regex = re.compile(f"({special_pattern})")

        parts = special_regex.split(text)
        ids = []
        for part in parts:
            if part in special_set:
                ids.append(self.special_tokens[part])
            elif part:
                for chunk in self.pat.findall(part):
                    chunk_bytes = chunk.encode("utf-8")
                    ids.extend(self._encode_chunk(chunk_bytes))
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode list of token IDs back into string.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded string.
        """
        byte_pieces: List[bytes] = []
        for idx in ids:
            if idx in self.vocab:
                byte_pieces.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                byte_pieces.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token ID: {idx}")

        full_bytes = b"".join(byte_pieces)
        return full_bytes.decode("utf-8", errors="replace")

    def save(
        self,
        vocab_filename: str | os.PathLike,
        merges_filename: str | os.PathLike,
    ) -> None:
        """
        Save vocabulary and merge rules to JSON/text files.
        """
        # Save vocabulary: ID -> hex string or latin1 string representation
        vocab_data = {
            str(idx): token.hex() for idx, token in self.vocab.items()
        }
        with open(vocab_filename, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2)

        # Save merges: pair of hex strings
        merges_data = [
            (p1.hex(), p2.hex()) for p1, p2 in self.merges
        ]
        with open(merges_filename, "w", encoding="utf-8") as f:
            json.dump({
                "merges": merges_data,
                "special_tokens": self.special_tokens,
            }, f, indent=2)

    @classmethod
    def from_files(
        cls,
        vocab_filename: str | os.PathLike,
        merges_filename: str | os.PathLike,
        special_tokens: Dict[str, int] | None = None,
    ) -> BPETokenizer:
        """
        Load BPETokenizer from vocabulary and merge files.
        """
        with open(vocab_filename, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        vocab: Dict[int, bytes] = {
            int(idx): bytes.fromhex(hex_str) for idx, hex_str in vocab_data.items()
        }

        with open(merges_filename, "r", encoding="utf-8") as f:
            merges_dict = json.load(f)

        merges = [
            (bytes.fromhex(p1), bytes.fromhex(p2)) for p1, p2 in merges_dict["merges"]
        ]
        loaded_special_tokens = merges_dict.get("special_tokens", {})
        if special_tokens:
            loaded_special_tokens.update(special_tokens)

        return cls(vocab=vocab, merges=merges, special_tokens=loaded_special_tokens)
