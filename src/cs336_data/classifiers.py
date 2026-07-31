"""
Language identification, toxic content, and text quality classifiers.
"""

import re
from typing import Optional

# List of common English words used for lightweight language identification fallback
ENGLISH_MARKER_WORDS = {
    "the", "and", "that", "have", "for", "not", "with", "you", "this", "but",
    "his", "from", "they", "say", "her", "she", "or", "an", "will", "my",
    "one", "all", "would", "there", "their", "what", "so", "up", "out", "if"
}

# List of toxic keywords for rule-based fallback detection
TOXIC_KEYWORDS = {
    "hate", "racist", "slur", "violence", "exploit", "harass", "nsfw", "porn"
}


class LanguageClassifier:
    """Classifier for detecting text language (e.g., English filtering)."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.fasttext_model = None
        if model_path:
            try:
                import fasttext
                self.fasttext_model = fasttext.load_model(model_path)
            except Exception:
                self.fasttext_model = None

    def predict_language(self, text: str) -> tuple[str, float]:
        """
        Predict the primary language of the text and the confidence score.
        
        Returns:
            Tuple of (language_code, confidence_score).
        """
        clean_text = text.replace("\n", " ").strip()
        if not clean_text:
            return ("unknown", 0.0)

        if self.fasttext_model is not None:
            labels, probs = self.fasttext_model.predict(clean_text)
            lang = labels[0].replace("__label__", "")
            prob = float(probs[0])
            return (lang, prob)

        # Fallback heuristic language identifier
        words = [w.lower().strip(".,!?\"'()") for w in clean_text.split()]
        if not words:
            return ("unknown", 0.0)

        english_matches = sum(1 for w in words if w in ENGLISH_MARKER_WORDS)
        score = min(1.0, english_matches / max(len(words) * 0.2, 1.0))
        
        if score > 0.3:
            return ("en", score)
        return ("other", 1.0 - score)

    def is_english(self, text: str, threshold: float = 0.5) -> bool:
        """Check if text language is English with confidence >= threshold."""
        lang, score = self.predict_language(text)
        return lang == "en" and score >= threshold


class ToxicContentClassifier:
    """Classifier for detecting toxic, harmful, or inappropriate text content."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.fasttext_model = None
        if model_path:
            try:
                import fasttext
                self.fasttext_model = fasttext.load_model(model_path)
            except Exception:
                self.fasttext_model = None

    def predict_toxicity(self, text: str) -> float:
        """
        Predict toxicity score of text (0.0 = clean, 1.0 = highly toxic).
        """
        clean_text = text.replace("\n", " ").strip()
        if not clean_text:
            return 0.0

        if self.fasttext_model is not None:
            labels, probs = self.fasttext_model.predict(clean_text)
            label = labels[0].replace("__label__", "")
            prob = float(probs[0])
            if label.lower() in ("toxic", "harmful", "nsfw"):
                return prob
            return 1.0 - prob

        # Heuristic toxic content score based on keyword match density
        words = [w.lower().strip(".,!?\"'()") for w in clean_text.split()]
        if not words:
            return 0.0

        toxic_matches = sum(1 for w in words if w in TOXIC_KEYWORDS)
        toxicity_score = min(1.0, toxic_matches / max(len(words) * 0.1, 1.0))
        return toxicity_score

    def is_toxic(self, text: str, threshold: float = 0.5) -> bool:
        """Check if text toxicity score exceeds threshold."""
        return self.predict_toxicity(text) >= threshold


class QualityClassifier:
    """Classifier for scoring document quality relative to high-quality reference corpora."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path

    def predict_quality(self, text: str) -> float:
        """
        Predict document quality score in range [0.0, 1.0].
        High score indicates high similarity to quality corpora (e.g. Wikipedia/Books).
        """
        words = text.split()
        if not words:
            return 0.0

        # Heuristic quality metric: proportion of alphabetic words and average length
        alpha_words = sum(1 for w in words if w.isalpha())
        alpha_ratio = alpha_words / len(words)

        mean_len = sum(len(w) for w in words) / len(words)
        len_score = 1.0 if 3.5 <= mean_len <= 8.5 else 0.5

        quality_score = min(1.0, alpha_ratio * 0.7 + len_score * 0.3)
        return quality_score

    def is_high_quality(self, text: str, threshold: float = 0.5) -> bool:
        """Check if text quality score meets or exceeds threshold."""
        return self.predict_quality(text) >= threshold
