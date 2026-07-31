"""
Heuristic quality filters for text content (Gopher paper heuristics).
"""

import re

# Standard set of common English stopwords
ENGLISH_STOPWORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what"
}


def check_word_count(text: str, min_words: int = 50, max_words: int = 100000) -> bool:
    """Check if document word count falls within [min_words, max_words]."""
    words = text.split()
    return min_words <= len(words) <= max_words


def check_mean_word_length(text: str, min_length: float = 3.0, max_length: float = 10.0) -> bool:
    """Check if mean word length falls within [min_length, max_length]."""
    words = text.split()
    if not words:
        return False
    mean_len = sum(len(word) for word in words) / len(words)
    return min_length <= mean_len <= max_length


def check_symbol_to_word_ratio(text: str, max_symbol_ratio: float = 0.1) -> bool:
    """Check if the ratio of symbol characters (#, ..., {}, [], <>) to total words is below threshold."""
    words = text.split()
    if not words:
        return False
    symbols = re.findall(r"[#{}<>\-\=\+\*]", text)
    ellipsis_count = len(re.findall(r"\.\.\.", text))
    symbol_ratio = (len(symbols) + ellipsis_count) / len(words)
    return symbol_ratio <= max_symbol_ratio


def check_bullet_point_lines(text: str, max_bullet_ratio: float = 0.9) -> bool:
    """Check if ratio of lines starting with bullet points (*, -, •) is below threshold."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    bullet_lines = sum(1 for line in lines if line.startswith(("*", "-", "•", "1.", "2.")))
    return (bullet_lines / len(lines)) <= max_bullet_ratio


def check_stopwords(text: str, min_stopwords: int = 2) -> bool:
    """Check if document contains at least min_stopwords English stop words."""
    words = [word.lower().strip(".,!?\"'()") for word in text.split()]
    count = sum(1 for word in words if word in ENGLISH_STOPWORDS)
    return count >= min_stopwords


def gopher_quality_filter(
    text: str,
    min_words: int = 50,
    max_words: int = 100000,
    min_mean_length: float = 3.0,
    max_mean_length: float = 10.0,
    max_symbol_ratio: float = 0.1,
    max_bullet_ratio: float = 0.9,
    min_stopwords: int = 2,
) -> bool:
    """
    Applies the full suite of Gopher quality filtering heuristics (Rae et al., 2021).
    
    Returns:
        True if the text passes all quality rules, False otherwise.
    """
    if not text or not text.strip():
        return False

    if not check_word_count(text, min_words, max_words):
        return False

    if not check_mean_word_length(text, min_mean_length, max_mean_length):
        return False

    if not check_symbol_to_word_ratio(text, max_symbol_ratio):
        return False

    if not check_bullet_point_lines(text, max_bullet_ratio):
        return False

    if not check_stopwords(text, min_stopwords):
        return False

    return True
