"""
This module provides robust parsing functions to extract structured answers 
from raw model outputs. All functions catch internal exceptions and return None 
if parsing fails, ensuring the evaluation pipeline never crashes on malformed output.
"""

from __future__ import annotations

import re


def extract_numeric(text: str) -> float | None:
    try:
        # Match integers, floats, and scientific notation (ignore commas)
        clean_text = str(text).replace(',', '')
        matches = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', clean_text)
        if matches:
            # Return the last number found (often the final answer)
            return float(matches[-1])
        return None
    except Exception:
        return None


def extract_path(text: str) -> str | None:
    try:
        # Match sequences of at least two tokens separated by commas or arrows
        # E.g., "A, B, C" or "A -> B -> C"
        pattern = r'\b[\w-]+(?:\s*(?:,|->)\s*[\w-]+)+\b'
        matches = re.findall(pattern, str(text))
        if not matches:
            return None
        
        # Take the last match block found
        last_match = matches[-1]
        
        # Split by comma or arrow, strip spaces, convert to upper case, join by comma
        tokens = re.split(r'\s*(?:,|->)\s*', last_match)
        normalized = ",".join(token.strip().upper() for token in tokens if token.strip())
        return normalized
    except Exception:
        return None


def extract_plan(text: str) -> list[str] | None:
    try:
        # Match pattern "move X from Y to Z" case-insensitively
        pattern = r'(?i)\bmove\s+[\w-]+\s+from\s+[\w-]+\s+to\s+[\w-]+\b'
        matches = re.findall(pattern, str(text))
        
        if not matches:
            return None
        
        moves = []
        for m in matches:
            # Replace multiple spaces with a single space and convert to lower case
            normalized = re.sub(r'\s+', ' ', m.strip().lower())
            moves.append(normalized)
            
        return moves
    except Exception:
        return None
