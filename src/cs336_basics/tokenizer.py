"""
Byte-Pair Encoding (BPE) Tokenizer implementation for Stanford CS336.
Supports GPT-2 regex pre-tokenization, training from raw text, encoding, decoding,
special token handling, and serialization/deserialization.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Iterable, List, Set, Tuple, Union

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

        self.pat = regex.compile(GPT2_SPLIT_REGEX)

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        special_tokens: List[str] | None = None,
    ) -> BPETokenizer:
        """
        Train BPE Tokenizer on raw text with fast incremental pair counting.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 to cover all byte values.")

        vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        
        if special_tokens:
            special_pattern = "|".join(re.escape(st) for st in special_tokens)
            text_chunks = re.split(special_pattern, text)
        else:
            text_chunks = [text]

        words = []
        for chunk in text_chunks:
            words.extend(regex.findall(GPT2_SPLIT_REGEX, chunk))

        # Count word frequencies
        raw_counts: Dict[Tuple[bytes, ...], int] = {}
        for word in words:
            byte_tuple = tuple(bytes([b]) for b in word.encode("utf-8"))
            raw_counts[byte_tuple] = raw_counts.get(byte_tuple, 0) + 1

        # Initial pair counts and inverted index (pair -> set of words containing pair)
        pair_counts: Dict[Tuple[bytes, bytes], int] = {}
        where_pair: Dict[Tuple[bytes, bytes], Set[Tuple[bytes, ...]]] = {}

        for word, freq in raw_counts.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
                if pair not in where_pair:
                    where_pair[pair] = set()
                where_pair[pair].add(word)

        word_counts = dict(raw_counts)
        merges: List[Tuple[bytes, bytes]] = []

        num_specials = len(special_tokens) if special_tokens else 0
        num_merges = vocab_size - 256 - num_specials
        for _ in range(num_merges):
            if not pair_counts:
                break

            # Find pair with max frequency (break ties lexicographically for reference BPE)
            best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
            if pair_counts[best_pair] == 0:
                break

            merged_token = best_pair[0] + best_pair[1]
            new_id = len(vocab)
            vocab[new_id] = merged_token
            merges.append(best_pair)

            # Get affected words
            affected_words = list(where_pair.get(best_pair, set()))

            for old_word in affected_words:
                freq = word_counts.pop(old_word)

                # Remove old pairs of this word
                for i in range(len(old_word) - 1):
                    p = (old_word[i], old_word[i + 1])
                    pair_counts[p] -= freq
                    if pair_counts[p] == 0:
                        del pair_counts[p]
                    if p in where_pair and old_word in where_pair[p]:
                        where_pair[p].remove(old_word)

                # Merge best_pair in old_word
                new_word_list: List[bytes] = []
                i = 0
                while i < len(old_word):
                    if i < len(old_word) - 1 and (old_word[i], old_word[i + 1]) == best_pair:
                        new_word_list.append(merged_token)
                        i += 2
                    else:
                        new_word_list.append(old_word[i])
                        i += 1
                new_word = tuple(new_word_list)

                # Add new_word to word_counts
                word_counts[new_word] = word_counts.get(new_word, 0) + freq

                # Add new pairs of new_word
                for i in range(len(new_word) - 1):
                    p = (new_word[i], new_word[i + 1])
                    pair_counts[p] = pair_counts.get(p, 0) + freq
                    if p not in where_pair:
                        where_pair[p] = set()
                    where_pair[p].add(new_word)

        # Construct special tokens map and insert into vocab
        special_token_map: Dict[str, int] = {}
        if special_tokens:
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                new_id = len(vocab)
                vocab[new_id] = token_bytes
                special_token_map[token] = new_id

        return cls(vocab=vocab, merges=merges, special_tokens=special_token_map)


    def _encode_chunk(self, chunk_bytes: bytes) -> List[int]:
        if not chunk_bytes:
            return []

        parts: List[bytes] = [bytes([b]) for b in chunk_bytes]

        while len(parts) >= 2:
            pairs = [(parts[i], parts[i + 1]) for i in range(len(parts) - 1)]
            best_pair = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            
            if best_pair not in self.bpe_ranks:
                break

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
        allowed_special: Union[Set[str], str] = "all",
    ) -> List[int]:
        if allowed_special == "all":
            special_set = set(self.special_tokens.keys())
        elif allowed_special == "none":
            special_set = set()
        elif isinstance(allowed_special, set):
            special_set = allowed_special
        else:
            special_set = set(self.special_tokens.keys())

        if not special_set:
            ids: List[int] = []
            for chunk in self.pat.findall(text):
                chunk_bytes = chunk.encode("utf-8")
                ids.extend(self._encode_chunk(chunk_bytes))
            return ids

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

    def encode_iterable(
        self,
        iterable: Iterable[str],
        allowed_special: Union[Set[str], str] = "all",
    ):
        for line in iterable:
            for token_id in self.encode(line, allowed_special=allowed_special):
                yield token_id

    def decode(self, ids: List[int]) -> str:
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
        vocab_data = {
            str(idx): token.hex() for idx, token in self.vocab.items()
        }
        with open(vocab_filename, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2)

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
