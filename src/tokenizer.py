"""
src/tokenizer.py - Neuro-Symbolic Topological Lattice (NSTL)
Domain-Agnostic Sub-Word and Identifier Tokenizer.
"""

from __future__ import annotations
import re
import sys
from typing import Set, FrozenSet

# Universal camelCase and snake_case boundary splitter
_IDENTIFIER_SPLIT_REGEX = re.compile(
    r'[_\-\s\.]+|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[a-zA-Z])(?=[0-9])|(?<=[0-9])(?=[a-zA-Z])'
)


class CellTokenizer:
    """Tokenizes code identifiers, cell IDs, and user prompts into sub-word tokens."""

    @classmethod
    def tokenize_identifier(cls, identifier: str) -> Set[str]:
        """
        Splits camelCase, snake_case, and dotted identifiers into distinct sub-tokens.
        """
        if not identifier:
            return set()

        parts = _IDENTIFIER_SPLIT_REGEX.split(identifier)
        tokens = set()

        for p in parts:
            p_clean = p.lower().strip()
            if len(p_clean) > 1:
                tokens.add(p_clean)

        clean_id = identifier.lower().strip()
        if clean_id:
            tokens.add(clean_id)
        return tokens

    @classmethod
    def tokenize_prompt(cls, prompt: str, remove_stopwords: bool = False) -> Set[str]:
        """Tokenizes user natural language prompts using word-boundary segmentation."""
        if not prompt:
            return set()

        raw_tokens = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', prompt.lower()))
        return {t for t in raw_tokens if len(t) > 1}

    @classmethod
    def tokenize_cell(cls, cell_id: str, keywords: Set[str]) -> Set[str]:
        """Produces a unified token set for a cell using its ID and declared keywords."""
        tokens = cls.tokenize_identifier(cell_id)
        for kw in keywords:
            tokens.update(cls.tokenize_identifier(kw))
        return tokens
