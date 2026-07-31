"""
Unit tests for HTML text extraction.
"""

import pytest
from cs336_data.extraction import extract_text_from_html


def test_extract_basic_html():
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>Test</title></head>
    <body>
        <h1>Hello World</h1>
        <p>This is a test paragraph for HTML text extraction.</p>
    </body>
    </html>
    """
    extracted = extract_text_from_html(html)
    assert "Hello World" in extracted
    assert "This is a test paragraph for HTML text extraction." in extracted


def test_strip_script_and_style():
    html = """
    <html>
    <body>
        <style>body { color: red; }</style>
        <p>Content to keep</p>
        <script>console.log("Remove me");</script>
    </body>
    </html>
    """
    extracted = extract_text_from_html(html)
    assert "Content to keep" in extracted
    assert "color: red" not in extracted
    assert "Remove me" not in extracted


def test_bytes_input():
    html_bytes = b"<html><body><p>Bytes content test</p></body></html>"
    extracted = extract_text_from_html(html_bytes)
    assert extracted == "Bytes content test"


def test_empty_html():
    assert extract_text_from_html("") == ""
    assert extract_text_from_html(b"") == ""
