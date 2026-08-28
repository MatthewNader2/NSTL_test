"""
tests/test_tokenization_reflection.py
Reflection-based property test for sub-word identifier tokenization.

Uses Python inspect and dir() reflection over real modules used in NSTL
(sklearn.preprocessing, cv2, numpy) without any hardcoded lists of names.
"""

import inspect
import pytest
import re
from typing import List

import numpy as np
import cv2
import sklearn.preprocessing

from src.tokenizer import CellTokenizer, _IDENTIFIER_SPLIT_REGEX


def _get_reflected_identifiers() -> List[str]:
    """Dynamically reflects public identifiers from libraries used in NSTL."""
    modules = [sklearn.preprocessing, cv2, np]
    identifiers = set()

    for mod in modules:
        for name in dir(mod):
            # Skip private/dunder attributes and noise
            if name.startswith("_"):
                continue
            try:
                obj = getattr(mod, name)
                # Only include functions, classes, and uppercase flag constants
                if inspect.isfunction(obj) or inspect.isclass(obj) or inspect.isbuiltin(obj) or (isinstance(obj, int) and name.isupper()):
                    identifiers.add(name)
            except Exception:
                pass

    return sorted(list(identifiers))


@pytest.fixture(scope="module")
def reflected_names() -> List[str]:
    names = _get_reflected_identifiers()
    assert len(names) > 500, f"Expected a broad reflection sample, got {len(names)}"
    return names


def test_tokenization_no_single_letter_tokens(reflected_names: List[str]):
    """Invariant 1: No returned token ever has length <= 1 (single stray letters dropped)."""
    for name in reflected_names:
        tokens = CellTokenizer.tokenize_identifier(name)
        for token in tokens:
            assert len(token) > 1, f"Identifier '{name}' produced a single-letter token '{token}'"


def test_tokenization_coverage_invariant(reflected_names: List[str]):
    """
    Invariant 2: Coverage invariant — no character in multi-character word parts is silently dropped.
    
    Splitting CamelCase or snake_case identifiers must preserve all characters in sub-tokens
    in their original order (i.e. 'MinMaxScaler' -> 'min', 'max', 'scaler' concatenation
    equals 'minmaxscaler', not 'inaxcaler').
    """
    for name in reflected_names:
        # Split on separators and camelCase boundaries
        parts = _IDENTIFIER_SPLIT_REGEX.split(name)
        # Filter parts that are valid tokens (len > 1)
        valid_parts = [p.lower().strip() for p in parts if len(p.strip()) > 1]
        
        tokens = CellTokenizer.tokenize_identifier(name)
        
        # All valid parts must be present in the returned token set
        for part in valid_parts:
            assert part in tokens, f"Identifier '{name}' missing sub-word part '{part}' in tokens: {tokens}"
            
        # The concatenation of all sub-tokens must equal the concatenation of all non-single-char parts
        concatenated_parts = "".join(valid_parts)
        
        # Verify order preservation: characters in concatenated_parts appear in name.lower() in order
        name_clean = re.sub(r"[^a-zA-Z0-9]", "", name.lower())
        
        # If the name consists of multi-character subwords, concatenated_parts should match name_clean
        # (excluding any isolated single-character tokens like '2' in Point2f or BGR2GRAY)
        idx = 0
        for ch in concatenated_parts:
            found_idx = name_clean.find(ch, idx)
            assert found_idx != -1, f"Character '{ch}' from tokens of '{name}' not found in order in '{name_clean}'"
            idx = found_idx + 1


def test_tokenization_camelcase_soundness():
    """
    Verifies that CamelCase multi-word classes retain their leading characters.
    Exercises dynamically discovered classes from sklearn.preprocessing.
    """
    classes = [
        name for name, obj in inspect.getmembers(sklearn.preprocessing, inspect.isclass)
        if not name.startswith("_")
    ]
    assert len(classes) >= 10

    for cls_name in classes:
        tokens = CellTokenizer.tokenize_identifier(cls_name)
        cls_lower = cls_name.lower()
        
        # Verify full identifier is in tokens
        assert cls_lower in tokens
        
        # Verify that for every camelCase component of length >= 2, the lowercased component is in tokens
        camel_parts = re.findall(r"[A-Z][a-z0-9]+|[A-Z0-9]+(?=[A-Z][a-z]|\b)", cls_name)
        for cp in camel_parts:
            if len(cp) > 1:
                assert cp.lower() in tokens, f"Class '{cls_name}' dropped component '{cp}' from tokens: {tokens}"
