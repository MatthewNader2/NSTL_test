"""
src/tokenizer.py - Neuro-Symbolic Topological Lattice (NSTL)
Domain-Agnostic Sub-Word and Identifier Tokenizer.

Deterministic, mathematically formal character-level transition scanner.
Zero regular expressions.

Includes a universal Porter-class suffix-stripping normalizer implemented as a
pure character-transition state machine (no regex, no lexicons, no domain
tables). The normalizer is applied IDENTICALLY to prompt tokens and identifier
tokens, so morphological variants ("testing"/"test", "training"/"train",
"regression"/"regressor", "normalized"/"normalize") collapse to the same
token regardless of which side of the match they appear on. This is a
language-level algorithm, not a domain vocabulary: it contains zero library,
intent, or keyword knowledge.
"""

from __future__ import annotations
from typing import Set, FrozenSet, List


_VOWELS = frozenset("aeiou")


def _is_vowel(ch: str) -> bool:
    return ch.lower() in _VOWELS


def _measure(stem: str) -> int:
    """
    Porter measure m: the number of VC sequences in <C?(VC){m}V?{0,2}>...
    Computed as the count of vowel-to-consonant transitions over the sequence
    of vowel/consonant classes of the word.
    """
    m = 0
    prev_vowel = False
    for ch in stem.lower():
        if ch.isalpha():
            v = _is_vowel(ch)
            if prev_vowel and not v:
                m += 1
            prev_vowel = v
    return m


def _contains_vowel(stem: str) -> bool:
    return any(_is_vowel(c) for c in stem)


def _ends_double_consonant(stem: str) -> bool:
    return (
        len(stem) >= 2
        and stem[-1] == stem[-2]
        and not _is_vowel(stem[-1])
        and stem[-1].isalpha()
    )


def _ends_cvc(stem: str) -> bool:
    if len(stem) < 3:
        return False
    a, b, c = stem[-3], stem[-2], stem[-1]
    if not (a.isalpha() and b.isalpha() and c.isalpha()):
        return False
    if _is_vowel(c):
        return False
    if b not in ("a", "e", "i", "o", "u") and b != "w" and b != "x" and b != "y":
        return False
    return not _is_vowel(a)


def normalize_token(token: str) -> str:
    """
    Universal Porter-class stem for a single lowercase alphabetic token.
    Deterministic state machine over suffix transitions; zero regex, zero
    lexicons. Words of length <= 2 are returned unchanged.
    """
    w = token.strip().lower()
    if not w.isalpha() or len(w) <= 2:
        return w

    # ---------- Step 1a: plurals ----------
    if w.endswith("sses") or w.endswith("ies"):
        w = w[:-2] if w.endswith("sses") else w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s") and len(w) > 3 and not w.endswith("us") and not w.endswith("is"):
        w = w[:-1]

    # ---------- Step 1b: -ed / -ing ----------
    flag_1b = False
    if w.endswith("eed"):
        if _measure(w[:-3]) > 0:
            w = w[:-1]
    elif w.endswith("ed"):
        if _contains_vowel(w[:-2]):
            w = w[:-2]
            flag_1b = True
    elif w.endswith("ing"):
        if _contains_vowel(w[:-3]):
            w = w[:-3]
            flag_1b = True

    if flag_1b:
        if w.endswith(("at", "bl", "iz")):
            w += "e"
        elif _ends_double_consonant(w) and w[-1] not in ("l", "s", "z"):
            w = w[:-1]
        elif _measure(w) == 1 and _ends_cvc(w):
            w += "e"

    # ---------- Step 1c: final y ----------
    if w.endswith("y") and _contains_vowel(w[:-1]):
        w = w[:-1] + "i"

    # ---------- Step 2: double suffix reductions (m > 0 on the stem) ----------
    step2 = (
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
        ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
        ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
        ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
        ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
    )
    for suffix, repl in step2:
        if w.endswith(suffix):
            stem = w[: -len(suffix)]
            if _measure(stem) > 0:
                w = stem + repl
            break

    # ---------- Step 3: -ic/-full/-ness etc. ----------
    step3 = (
        ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
        ("ical", "ic"), ("ful", ""), ("ness", ""),
    )
    for suffix, repl in step3:
        if w.endswith(suffix):
            stem = w[: -len(suffix)]
            if _measure(stem) > 0:
                w = stem + repl
            break

    # ---------- Step 4: residual suffixes (m > 1) ----------
    step4 = (
        "al", "ance", "ence", "er", "or", "ic", "able", "ible", "ant", "ement",
        "ment", "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
    )
    for suffix in step4:
        if w.endswith(suffix):
            stem = w[: -len(suffix)]
            if _measure(stem) > 1:
                if suffix == "ion" and stem and stem[-1] not in ("s", "t"):
                    break  # ion only after s/t
                w = stem
            break

    # ---------- Step 5a: final silent e ----------
    if w.endswith("e"):
        stem = w[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _ends_cvc(stem)):
            w = stem

    # ---------- Step 5b: double l ----------
    if w.endswith("ll") and _measure(w) > 1:
        w = w[:-1]

    return w


class CellTokenizer:
    """Tokenizes code identifiers, cell IDs, and user prompts into sub-word tokens without regular expressions."""

    @classmethod
    def tokenize_identifier(cls, identifier: str) -> Set[str]:
        """
        Splits camelCase, snake_case, kebab-case, and dotted identifiers into distinct sub-tokens
        using a deterministic character-level transition scanner. Each emitted token is paired
        with its morphological stem so cross-side variant matches succeed symmetrically.
        """
        if not identifier:
            return set()

        clean_id = str(identifier).strip()
        if not clean_id:
            return set()

        tokens: Set[str] = set()
        n = len(clean_id)
        current_chunk: List[str] = []

        def _emit(chunk: List[str]) -> None:
            t = "".join(chunk).lower().strip()
            if len(t) > 1:
                tokens.add(t)
                stem = normalize_token(t)
                if len(stem) > 1:
                    tokens.add(stem)

        # Mathematical character transition state machine:
        # Boundaries occur at:
        # 1. Non-alphanumeric punctuation (_, -, ., space, etc.)
        # 2. Lowercase followed by Uppercase (fooBar -> foo, Bar)
        # 3. Uppercase followed by Uppercase then Lowercase (HTTPServer -> HTTP, Server)
        # 4. Letter followed by Digit or Digit followed by Letter
        for i, ch in enumerate(clean_id):
            if not ch.isalnum():
                if current_chunk:
                    _emit(current_chunk)
                    current_chunk = []
                continue

            if current_chunk:
                prev = current_chunk[-1]
                # Lowercase to Uppercase transition
                if prev.islower() and ch.isupper():
                    _emit(current_chunk)
                    current_chunk = [ch]
                    continue
                # Uppercase to Uppercase followed by Lowercase (e.g. 'P' in 'HTTPServer' followed by 'e')
                if prev.isupper() and ch.isupper() and i + 1 < n and clean_id[i + 1].islower():
                    _emit(current_chunk)
                    current_chunk = [ch]
                    continue
                # Letter to Digit
                if prev.isalpha() and ch.isdigit():
                    _emit(current_chunk)
                    current_chunk = [ch]
                    continue
                # Digit to Letter
                if prev.isdigit() and ch.isalpha():
                    _emit(current_chunk)
                    current_chunk = [ch]
                    continue

            current_chunk.append(ch)

        if current_chunk:
            _emit(current_chunk)

        clean_full = "".join(c for c in clean_id if c.isalnum() or c in ("_", "-")).lower().strip()
        if len(clean_full) > 1:
            tokens.add(clean_full)

        return tokens

    @classmethod
    def tokenize_prompt(cls, prompt: str, remove_stopwords: bool = False) -> Set[str]:
        """
        Tokenizes user natural language prompts using word-boundary segmentation.
        Zero regular expressions: scans alphanumeric sequences. Each word token is
        paired with its morphological stem (symmetric with identifier tokenization).
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
                        if w.isalpha():
                            stem = normalize_token(w)
                            if len(stem) > 1:
                                tokens.add(stem)
                    current_word = []

        if current_word:
            w = "".join(current_word)
            if len(w) > 1 and (w[0].isalpha() or w[0] == '_'):
                tokens.add(w)
                if w.isalpha():
                    stem = normalize_token(w)
                    if len(stem) > 1:
                        tokens.add(stem)

        return tokens

    @classmethod
    def tokenize_cell(cls, cell_id: str, keywords: Set[str]) -> Set[str]:
        """Produces a unified token set for a cell using its ID and declared keywords."""
        tokens = cls.tokenize_identifier(cell_id)
        for kw in keywords:
            tokens.update(cls.tokenize_identifier(kw))
        return tokens
