"""
HTML text extraction utilities for Common Crawl and web pages.
"""

from bs4 import BeautifulSoup
import re


def extract_text_from_html(html_content: str | bytes) -> str:
    """
    Extract clean plain text from HTML content.
    
    Removes script, style, header, footer, navigation, and form tags while
    preserving main textual content and paragraph/line breaks.
    
    Args:
        html_content: HTML content as string or bytes.
        
    Returns:
        Cleaned plain text extracted from HTML.
    """
    if isinstance(html_content, bytes):
        try:
            html_str = html_content.decode("utf-8")
        except UnicodeDecodeError:
            html_str = html_content.decode("latin-1", errors="ignore")
    else:
        html_str = html_content

    if not html_str.strip():
        return ""

    soup = BeautifulSoup(html_str, "html.parser")

    # Remove non-textual or boilerplate elements
    for element in soup(["script", "style", "header", "footer", "nav", "noscript", "form", "svg", "iframe"]):
        element.decompose()

    # Extract text with newline separator for block elements
    text = soup.get_text(separator="\n")

    # Clean whitespace line by line
    lines = [line.strip() for line in text.splitlines()]
    
    # Remove empty lines while maintaining double newlines between paragraphs
    cleaned_lines = []
    prev_empty = False
    for line in lines:
        if line:
            cleaned_lines.append(line)
            prev_empty = False
        elif not prev_empty:
            cleaned_lines.append("")
            prev_empty = True

    result = "\n".join(cleaned_lines).strip()
    return result
