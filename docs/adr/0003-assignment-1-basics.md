# ADR 0003: Scope and Architecture for Assignment 1 (Language Modeling Basics)

* **Status:** Accepted
* **Date:** 2026-07-31

## Context

Assignment 1 ("Basics") of Stanford CS336 requires implementing a complete language modeling pipeline from scratch, including:
1. A Byte-Pair Encoding (BPE) Tokenizer.
2. Transformer model building blocks (RMSNorm, SwiGLU, RoPE, Multi-Head Causal Attention) and full language model.
3. Cross-entropy loss, custom AdamW optimizer, learning rate scheduler, and gradient clipping.
4. Data processing, training loop, evaluation (perplexity), and model checkpointing.

## Decision

1. **Module Organization (`src/cs336_basics/`):**
   - `src/cs336_basics/__init__.py`: Package initialization.
   - `src/cs336_basics/tokenizer.py`: BPE tokenizer training, encoding, decoding, pre-tokenization regex splitting, and special token handling.
   - `src/cs336_basics/model.py`: Transformer architectural components (`RMSNorm`, `SwiGLU`, `RotaryPositionalEmbedding`, `CausalSelfAttention`, `TransformerBlock`, `TransformerLM`).
   - `src/cs336_basics/optimizer.py`: Custom `AdamW` implementation and learning rate scheduler (warmup + cosine decay) with gradient clipping.
   - `src/cs336_basics/loss.py`: Custom numerically stable `cross_entropy` loss function.
   - `src/cs336_basics/dataset.py`: Memory-mapped / buffered data loader for training batches.
   - `src/cs336_basics/train.py`: Training loop, evaluation, checkpointing, and perplexity calculation.

2. **Development & Testing Methodology:**
   - Tests will be placed in `tests/` with individual unit tests for each component (`test_tokenizer.py`, `test_model.py`, `test_optimizer.py`, `test_loss.py`, `test_train.py`).
   - Temporary test scripts or benchmark runs will be created in `tmp/` during development and cleaned up upon execution per workspace guidelines.

3. **Dependencies:**
   - Standard Python libraries plus PyTorch, NumPy, regex, pytest, and tqdm in a local virtual environment (`.venv`).

## Consequences

- Highly modular codebase allowing step-by-step verification of each building block.
- Solid base for subsequent assignments (scaling, parallel training, efficiency optimizations).
