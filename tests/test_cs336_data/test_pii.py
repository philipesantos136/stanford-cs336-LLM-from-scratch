"""
Unit tests for PII masking.
"""

from cs336_data.pii import mask_emails, mask_phone_numbers, mask_ip_addresses


def test_mask_emails():
    text = "Contact us at alice@example.com or bob.smith@work-domain.org for support."
    masked, count = mask_emails(text)
    assert count == 2
    assert "alice@example.com" not in masked
    assert "bob.smith@work-domain.org" not in masked
    assert masked.count("|||EMAIL_ADDRESS|||") == 2


def test_mask_phone_numbers():
    text = "Call +1 555-123-4567 or (555) 987-6543 today."
    masked, count = mask_phone_numbers(text)
    assert count >= 1
    assert "|||PHONE_NUMBER|||" in masked


def test_mask_ip_addresses():
    text = "Server connects from 192.168.1.100 and gateway 10.0.0.1."
    masked, count = mask_ip_addresses(text)
    assert count == 2
    assert "192.168.1.100" not in masked
    assert "10.0.0.1" not in masked
    assert masked.count("|||IP_ADDRESS|||") == 2
