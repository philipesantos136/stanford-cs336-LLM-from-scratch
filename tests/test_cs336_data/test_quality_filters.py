"""
Unit tests for Gopher quality filtering heuristics.
"""

from cs336_data.quality_filters import (
    gopher_quality_filter,
    check_word_count,
    check_mean_word_length,
    check_symbol_to_word_ratio,
    check_bullet_point_lines,
    check_stopwords,
)


def test_valid_document_passes_gopher():
    valid_doc = (
        "Language modeling is a core task in artificial intelligence and natural language processing. "
        "In this assignment, we implement data filtering and deduplication pipelines to train large "
        "language models from scratch. High quality training data is critical for achieving optimal "
        "perplexity and performance. We curate Common Crawl HTML pages using Gopher quality heuristics, "
        "MinHash locality sensitive hashing, and fastText language identification classifiers."
    )
    assert gopher_quality_filter(valid_doc, min_words=10) is True


def test_short_document_fails():
    short_doc = "Too short text."
    assert gopher_quality_filter(short_doc, min_words=50) is False


def test_symbol_heavy_document_fails():
    symbol_doc = "word " + "# # # < > { } [ ] " * 20
    assert check_symbol_to_word_ratio(symbol_doc, max_symbol_ratio=0.1) is False


def test_bullet_points_filter():
    bullet_doc = "* Line 1\n* Line 2\n* Line 3\n* Line 4\n* Line 5"
    assert check_bullet_point_lines(bullet_doc, max_bullet_ratio=0.8) is False


def test_stopwords_filter():
    no_stopwords = "Xylophone quartz cryptography cipher binary matrix tensor vector"
    assert check_stopwords(no_stopwords, min_stopwords=2) is False
