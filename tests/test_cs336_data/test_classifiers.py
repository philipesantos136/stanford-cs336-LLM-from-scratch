"""
Unit tests for language identification and content classifiers.
"""

from cs336_data.classifiers import (
    LanguageClassifier,
    ToxicContentClassifier,
    QualityClassifier,
)


def test_language_classifier():
    clf = LanguageClassifier()
    text_en = "This is a clean English paragraph with common words and normal structure for pretraining."
    assert clf.is_english(text_en) is True


def test_toxic_content_classifier():
    clf = ToxicContentClassifier()
    clean_text = "The machine learning algorithm optimizes the loss function during training iterations."
    assert clf.is_toxic(clean_text) is False


def test_quality_classifier():
    clf = QualityClassifier()
    high_quality_text = "The quick brown fox jumps over the lazy dog in standard written English prose."
    assert clf.is_high_quality(high_quality_text) is True
