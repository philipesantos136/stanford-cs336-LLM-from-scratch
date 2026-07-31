"""
PII (Personally Identifiable Information) masking utilities.
"""

import re

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
# Matches standard US and international phone number formats (e.g. +1 555-123-4567, (555) 123-4567, 555-123-4567)
PHONE_REGEX = re.compile(
    r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)
IPV4_REGEX = re.compile(
    r"\b(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
)
IPV6_REGEX = re.compile(
    r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
)


def mask_emails(text: str, replacement: str = "|||EMAIL_ADDRESS|||") -> tuple[str, int]:
    """
    Mask email addresses in text.
    
    Args:
        text: Input string.
        replacement: String placeholder to replace email addresses with.
        
    Returns:
        Tuple of (masked_text, count_of_replacements).
    """
    matches = EMAIL_REGEX.findall(text)
    count = len(matches)
    masked_text = EMAIL_REGEX.sub(replacement, text)
    return masked_text, count


def mask_phone_numbers(text: str, replacement: str = "|||PHONE_NUMBER|||") -> tuple[str, int]:
    """
    Mask phone numbers in text.
    
    Args:
        text: Input string.
        replacement: String placeholder to replace phone numbers with.
        
    Returns:
        Tuple of (masked_text, count_of_replacements).
    """
    matches = PHONE_REGEX.findall(text)
    count = len(matches)
    masked_text = PHONE_REGEX.sub(replacement, text)
    return masked_text, count


def mask_ip_addresses(text: str, replacement: str = "|||IP_ADDRESS|||") -> tuple[str, int]:
    """
    Mask IPv4 and IPv6 addresses in text.
    
    Args:
        text: Input string.
        replacement: String placeholder to replace IP addresses with.
        
    Returns:
        Tuple of (masked_text, count_of_replacements).
    """
    count = 0
    
    # Mask IPv4
    ipv4_matches = IPV4_REGEX.findall(text)
    count += len(ipv4_matches)
    masked_text = IPV4_REGEX.sub(replacement, text)
    
    # Mask IPv6
    ipv6_matches = IPV6_REGEX.findall(masked_text)
    count += len(ipv6_matches)
    masked_text = IPV6_REGEX.sub(replacement, masked_text)
    
    return masked_text, count
