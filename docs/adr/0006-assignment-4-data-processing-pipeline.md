# ADR 0006: Pretraining Data Processing Pipeline Architecture

## Status
Accepted

## Context
Training large language models (LLMs) from raw web crawls (e.g., Common Crawl) presents significant challenges:
1. **Low-Quality HTML Boilerplate**: Raw web pages contain navigation bars, ads, script tags, and HTML layout tags.
2. **Privacy Risks**: Web pages frequently contain Personally Identifiable Information (PII) such as email addresses, phone numbers, and IP addresses.
3. **Low-Quality & Toxic Text**: Spam, machine-generated junk, symbol-heavy pages, and harmful text impair model performance and alignment.
4. **Data Redundancy**: Near-duplicate pages and repeated boilerplate lines waste GPU compute during training and encourage memorization.

To solve these problems, **Assignment 4** introduces a modular pretraining data curation pipeline under package `cs336_data`.

## Decisions

### 1. HTML Text Extraction (`cs336_data.extraction`)
- Use BeautifulSoup parsing (`html.parser`) to decompose non-textual structural elements (`<script>`, `<style>`, `<header>`, `<footer>`, `<nav>`, `<noscript>`, `<form>`).
- Preserve structural paragraph breaks while stripping inline whitespace.

### 2. Regex-Based PII Masking (`cs336_data.pii`)
- Implement deterministic regular expressions to replace email addresses, phone numbers, and IPv4/IPv6 addresses with standard tokens: `|||EMAIL_ADDRESS|||`, `|||PHONE_NUMBER|||`, and `|||IP_ADDRESS|||`.

### 3. DeepMind Gopher Quality Filtering (`cs336_data.quality_filters`)
- Adopt the heuristic quality rules established in the DeepMind Gopher paper (Rae et al., 2021):
  - Word count within range $[50, 100000]$.
  - Mean word length within $[3.0, 10.0]$ characters.
  - Symbol character and ellipsis ratio below $0.1$.
  - Bullet point line ratio below $0.9$.
  - Presence of at least 2 common English stopwords.

### 4. Classification & Language Filtering (`cs336_data.classifiers`)
- Provide `LanguageClassifier`, `ToxicContentClassifier`, and `QualityClassifier` with fastText integration support and robust heuristic fallbacks for offline execution.

### 5. Exact & MinHash LSH Fuzzy Deduplication (`cs336_data.deduplication`)
- **Exact Line Deduplication**: Count global line occurrences across documents and strip non-unique repeated boilerplate lines.
- **MinHash + Locality Sensitive Hashing (LSH)**:
  - Extract word $k$-shingles ($k=5$).
  - Compute $K=64$ permuted linear hash functions $h_i(x) = (a_i \cdot x + b_i) \pmod{2^{31}-1}$.
  - Divide signatures into $B=16$ bands with $R=4$ rows per band.
  - Group documents sharing LSH buckets into candidate pairs, verify exact Jaccard similarity $\ge 0.7$, and extract connected components via Union-Find.

## Consequences
- Clean, structured pipeline ready for pretraining data preprocessing.
- High speed and low memory overhead for filtering billions of web tokens.
- Deterministic automated unit tests ensuring high code reliability.
