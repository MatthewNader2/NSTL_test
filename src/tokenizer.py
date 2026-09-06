"""
src/tokenizer.py - Neuro-Symbolic Topological Lattice (NSTL)
Domain-Agnostic Sub-Word and Identifier Tokenizer.

Deterministic, mathematically formal character-level transition scanner.
Zero regular expressions.
"""

from __future__ import annotations
import sys
from typing import Set, FrozenSet, List


class CellTokenizer:
    """Tokenizes code identifiers, cell IDs, and user prompts into sub-word tokens without regular expressions."""

    @classmethod
    def tokenize_identifier(cls, identifier: str) -> Set[str]:
        """
        Splits camelCase, snake_case, kebab-case, and dotted identifiers into distinct sub-tokens
        using a deterministic character-level transition scanner.
        """
        if not identifier:
            return set()

        clean_id = str(identifier).strip()
        if not clean_id:
            return set()

        tokens: Set[str] = set()
        n = len(clean_id)
        current_chunk: List[str] = []

        # Mathematical character transition state machine:
        # Boundaries occur at:
        # 1. Non-alphanumeric punctuation (_, -, ., space, etc.)
        # 2. Lowercase followed by Uppercase (fooBar -> foo, Bar)
        # 3. Uppercase followed by Uppercase then Lowercase (HTTPServer -> HTTP, Server)
        # 4. Letter followed by Digit or Digit followed by Letter
        for i, ch in enumerate(clean_id):
            if not ch.isalnum():
                if current_chunk:
                    t = "".join(current_chunk).lower().strip()
                    if len(t) > 1:
                        tokens.add(t)
                    current_chunk = []
                continue

            if current_chunk:
                prev = current_chunk[-1]
                # Lowercase to Uppercase transition
                if prev.islower() and ch.isupper():
                    t = "".join(current_chunk).lower().strip()
                    if len(t) > 1:
                        tokens.add(t)
                    current_chunk = [ch]
                    continue
                # Uppercase to Uppercase followed by Lowercase (e.g. 'P' in 'HTTPServer' followed by 'e')
                if prev.isupper() and ch.isupper() and i + 1 < n and clean_id[i + 1].islower():
                    t = "".join(current_chunk).lower().strip()
                    if len(t) > 1:
                        tokens.add(t)
                    current_chunk = [ch]
                    continue
                # Letter to Digit
                if prev.isalpha() and ch.isdigit():
                    t = "".join(current_chunk).lower().strip()
                    if len(t) > 1:
                        tokens.add(t)
                    current_chunk = [ch]
                    continue
                # Digit to Letter
                if prev.isdigit() and ch.isalpha():
                    t = "".join(current_chunk).lower().strip()
                    if len(t) > 1:
                        tokens.add(t)
                    current_chunk = [ch]
                    continue

            current_chunk.append(ch)

        if current_chunk:
            t = "".join(current_chunk).lower().strip()
            if len(t) > 1:
                tokens.add(t)

        clean_full = "".join(c for c in clean_id if c.isalnum() or c in ("_", "-")).lower().strip()
        if len(clean_full) > 1:
            tokens.add(clean_full)

        return tokens

    @classmethod
    def tokenize_prompt(cls, prompt: str, remove_stopwords: bool = False) -> Set[str]:
        """
        Tokenizes user natural language prompts using word-boundary segmentation.
        Zero regular expressions: scans alphanumeric sequences.
        """
        if not prompt:
            return set()

        tokens: Set[str] = set()
        current_word: List[str] = []

        for ch in prompt.lower():
            if ch.isalnum() or ch == '_':
                current_word.append(ch)
            else:
                if current_word:
                    w = "".join(current_word)
                    if len(w) > 1 and (w[0].isalpha() or w[0] == '_'):
                        tokens.add(w)
                    current_word = []

        if current_word:
            w = "".join(current_word)
            if len(w) > 1 and (w[0].isalpha() or w[0] == '_'):
                tokens.add(w)

        return tokens

    @classmethod
    def tokenize_cell(cls, cell_id: str, keywords: Set[str]) -> Set[str]:
        """Produces a unified token set for a cell using its ID and declared keywords."""
        tokens = cls.tokenize_identifier(cell_id)
        for kw in keywords:
            tokens.update(cls.tokenize_identifier(kw))
        return tokens
