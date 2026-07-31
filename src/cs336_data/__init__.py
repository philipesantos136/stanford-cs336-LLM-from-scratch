"""
cs336_data: Pretraining Data Curation and Processing Pipeline for LLMs.

Includes:
- HTML text extraction
- PII masking (emails, phone numbers, IP addresses)
- Gopher quality filters
- Language and content classifiers
- Exact line and MinHash/LSH document deduplication
"""

from .extraction import extract_text_from_html
from .pii import mask_emails, mask_phone_numbers, mask_ip_addresses
from .quality_filters import gopher_quality_filter
from .classifiers import LanguageClassifier, ToxicContentClassifier, QualityClassifier
from .deduplication import exact_line_deduplication, MinHashLSH

__all__ = [
    "extract_text_from_html",
    "mask_emails",
    "mask_phone_numbers",
    "mask_ip_addresses",
    "gopher_quality_filter",
    "LanguageClassifier",
    "ToxicContentClassifier",
    "QualityClassifier",
    "exact_line_deduplication",
    "MinHashLSH",
]
