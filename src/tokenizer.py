"""
src/tokenizer.py - Neuro-Symbolic Topological Lattice (NSTL)
Domain-Agnostic Sub-Word and Identifier Tokenizer.

Deterministic, mathematically formal character-level transition scanner.
Zero regular expressions.

Morphological normalization is provided by an OPT-IN English Porter-class
suffix stripper. It is applied IDENTICALLY to prompt tokens and identifier
tokens when enabled, so English morphological variants ("testing"/"test",
"training"/"train", "regression"/"regressor", "normalized"/"normalize")
collapse to the same token. The stemmer is (a) gated by the module-level
`ENABLE_ENGLISH_STEMMING` flag and (b) applied only to purely ASCII-alphabetic
tokens. Non-Latin scripts and any token containing digits or symbols pass
through unchanged. This is an English-language heuristic, not a universal
normalizer; disabling the flag yields a purely character-level tokenizer.
"""

from __future__ import annotations
import functools
from typing import Set, FrozenSet, List


# --------------------------------------------------------------------------- #
# Opt-in English morphological normalizer
# --------------------------------------------------------------------------- #

#: Set to False to disable English Porter-class stemming entirely. When
#: disabled, tokens are emitted verbatim (lowercased) with no suffix stripping.
ENABLE_ENGLISH_STEMMING: bool = True


_VOWELS = frozenset("aeiou")


def _is_vowel(ch: str) -> bool:
    return ch.lower() in _VOWELS


def _is_ascii_alpha_token(s: str) -> bool:
    """True iff every character of `s` is an ASCII letter (a-z / A-Z)."""
    if not s:
        return False
    for c in s:
        if not (c.isascii() and c.isalpha()):
            return False
    return True


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


@functools.lru_cache(maxsize=32768)
def normalize_token(token: str) -> str:
    """
    Opt-in English Porter-class stem. Applied ONLY to ASCII-alphabetic tokens
    when `ENABLE_ENGLISH_STEMMING` is True. All other tokens (non-Latin scripts,
    digits, symbols, mixed identifiers) are returned lowercased but unchanged.

    Deterministic state machine over suffix transitions; zero regex, zero
    lexicons. Tokens of length <= 2 are returned unchanged.
    """
    w = token.strip().lower()
    if len(w) <= 2:
        return w

    # Gate 1: global opt-in.
    if not ENABLE_ENGLISH_STEMMING:
        return w

    # Gate 2: English suffix rules only apply to ASCII-alphabetic tokens.
    if not _is_ascii_alpha_token(w):
        return w

    # ---------- Step 1a: plurals ----------
    if w.endswith("sses") or w.endswith("ies"):
        w = w[:-2]
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


# (apply_fixes_v7) Alias expansion. Tree JSON declares aliases; tokenizer
# expands them in both directions.
_ALIASES: dict = {}
_ALIASES_REV: dict = {}


def register_aliases(mapping: dict) -> None:
    global _ALIASES, _ALIASES_REV
    clean = {
        str(k).strip().lower(): str(v).strip().lower()
        for k, v in (mapping or {}).items()
        if k and v
    }
    if clean == _ALIASES:
        return
    _ALIASES = clean
    rev: dict = {}
    for k, v in _ALIASES.items():
        rev.setdefault(v, set()).add(k)
    _ALIASES_REV = rev
    try:
        _tokenize_identifier_cached.cache_clear()
    except Exception:
        pass


def _expand_aliases(tokens):
    if not _ALIASES or not tokens:
        return tokens
    out = set(tokens)
    for t in tokens:
        c = _ALIASES.get(t)
        if c:
            out.add(c)
        r = _ALIASES_REV.get(t)
        if r:
            out.update(r)
    return out


def _tokenize_identifier_cached(identifier: str) -> FrozenSet[str]:
    clean = str(identifier).strip()
    if not clean:
        return frozenset()
    tokens = set()

    def _emit(chunk):
        t = "".join(chunk).lower().strip()
        if len(t) > 1:
            tokens.add(t)
            s = normalize_token(t)
            if len(s) > 1:
                tokens.add(s)

    seg, chunk = [], []
    for ch in clean:
        if ch.isalnum():
            seg.append(ch)
        elif seg:
            _emit(seg)
            seg = []
    if seg:
        _emit(seg)

    n = len(clean)
    for i, ch in enumerate(clean):
        if not ch.isalnum():
            if chunk:
                _emit(chunk)
                chunk = []
            continue
        if chunk:
            p = chunk[-1]
            if (
                (p.islower() and ch.isupper())
                or (p.isupper() and ch.isupper() and i + 1 < n and clean[i + 1].islower())
                or (p.isalpha() and ch.isdigit())
                or (p.isdigit() and ch.isalpha())
            ):
                _emit(chunk)
                chunk = [ch]
                continue
        chunk.append(ch)
    if chunk:
        _emit(chunk)

    full = "".join(c for c in clean if c.isalnum() or c in ("_", "-")).lower().strip()
    if len(full) > 1:
        tokens.add(full)

    if _ALIASES:  # (apply_fixes_v7)
        tokens = _expand_aliases(tokens)

    return frozenset(tokens)


class CellTokenizer:
    """Tokenizes code identifiers, cell IDs, and user prompts into sub-word tokens without regular expressions."""

    @classmethod
    def tokenize_identifier(cls, identifier: str) -> Set[str]:
        """
        Splits camelCase, snake_case, kebab-case, and dotted identifiers into distinct sub-tokens
        using a deterministic character-level transition scanner. Each emitted token is paired
        with its morphological stem (when the English stemmer is enabled) so cross-side variant
        matches succeed symmetrically.
        """
        if not identifier:
            return set()
        return set(_tokenize_identifier_cached(str(identifier)))

    @classmethod
    def tokenize_identifier_frozen(cls, identifier: str) -> FrozenSet[str]:
        """Cached immutable variant returning FrozenSet[str]."""
        if not identifier:
            return frozenset()
        return _tokenize_identifier_cached(str(identifier))

    @classmethod
    def tokenize_prompt(cls, prompt: str, remove_stopwords: bool = False) -> Set[str]:
        if not prompt:
            return set()
        tokens = set()
        cur = []
        for ch in prompt.lower():
            if ch.isalnum() or ch == '_':
                cur.append(ch)
            else:
                if cur:
                    w = "".join(cur)
                    if len(w) > 1 and (w[0].isalpha() or w[0] == '_'):
                        tokens.add(w)
                        if w.isalpha():
                            s = normalize_token(w)
                            if len(s) > 1:
                                tokens.add(s)
                    cur = []
        if cur:
            w = "".join(cur)
            if len(w) > 1 and (w[0].isalpha() or w[0] == '_'):
                tokens.add(w)
                if w.isalpha():
                    s = normalize_token(w)
                    if len(s) > 1:
                        tokens.add(s)
        for raw in prompt.split():
            cw = raw.strip(",;.:!?()[]{}\'\"")
            if any(c.isupper() for c in cw) or '_' in cw or '.' in cw or any(c.isdigit() for c in cw):
                tokens.update(cls.tokenize_identifier(cw))
        if _ALIASES:  # (apply_fixes_v7)
            tokens = _expand_aliases(tokens)
        return tokens

    @classmethod
    def tokenize_cell(cls, cell_id: str, keywords: Set[str]) -> Set[str]:
        """Produces a unified token set for a cell using its ID and declared keywords."""
        tokens = cls.tokenize_identifier(cell_id)
        for kw in keywords:
            tokens.update(cls.tokenize_identifier(kw))
        return tokens

    @classmethod
    def split_prompt_clauses(cls, prompt: str) -> List[str]:
        """
        Splits a prompt into logical action clauses without breaking inside quotes,
        brackets, filter expressions (e.g. 'x > 10 and y < 20'), or compound nouns.
        Deterministic, syntax-aware, zero external dependencies.

        Verb-anchored splits are driven EXCLUSIVELY by the TypeRegistry operation
        token pools; there is no hardcoded data-science verb list. When the
        TypeRegistry is unavailable, only punctuation-based and connective-word
        splits apply.
        """
        if not prompt or not prompt.strip():
            return []

        text = prompt.strip()
        # 1. Mask quoted strings and bracketed expressions
        protected = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch in ("'", '"'):
                j = i + 1
                while j < n and text[j] != ch:
                    if text[j] == '\\':
                        j += 1
                    j += 1
                if j < n:
                    protected.append((i, j + 1))
                    i = j + 1
                    continue
            elif ch in ('(', '['):
                closing = ')' if ch == '(' else ']'
                depth = 1
                j = i + 1
                while j < n and depth > 0:
                    if text[j] == ch:
                        depth += 1
                    elif text[j] == closing:
                        depth -= 1
                    j += 1
                protected.append((i, j))
                i = j
                continue
            i += 1

        def _is_inside_protected(idx: int) -> bool:
            return any(s <= idx < e for s, e in protected)

        # 2. Tokenize words with positions
        raw_words = []
        cur = []
        w_start = None
        for idx, ch in enumerate(text):
            if not ch.isspace():
                if w_start is None:
                    w_start = idx
                cur.append(ch)
            else:
                if cur:
                    raw_words.append((w_start, idx, "".join(cur)))
                    cur = []
                    w_start = None
        if cur and w_start is not None:
            raw_words.append((w_start, len(text), "".join(cur)))

        # Resolve target operation verbs dynamically from TypeRegistry.
        try:
            from .lattice import TypeRegistry
        except (ImportError, ValueError):
            try:
                from lattice import TypeRegistry
            except (ImportError, ValueError):
                TypeRegistry = None

        reg = TypeRegistry.get_instance() if TypeRegistry is not None else None
        if reg is not None:
            target_verbs = (
                reg.get_operation_tokens()
                | reg.get_estimator_verbs()
                | reg.get_egress_tokens()
            )
        else:
            target_verbs = set()

        # 3. Detect clause boundaries
        split_positions = set()
        for idx, (s, e, w) in enumerate(raw_words):
            if _is_inside_protected(s) and _is_inside_protected(e - 1):
                continue
            clean = w.strip(".,;:!?")
            clean_lower = clean.lower()

            if ";" in w and not (_is_inside_protected(s) and _is_inside_protected(e - 1)):
                split_positions.add(e)
                continue

            if w.endswith(".") and idx + 1 < len(raw_words) and not (_is_inside_protected(s) and _is_inside_protected(e - 1)):
                next_w = raw_words[idx + 1][2]
                if next_w and next_w[0].isupper():
                    split_positions.add(e)
                    continue

            if clean_lower in ("then", "next", "afterwards"):
                split_positions.add(s)
                continue

            if clean_lower == "and":
                in_filter = False
                for back in range(max(0, idx - 4), idx):
                    bw = raw_words[back][2].lower()
                    if any(op in bw for op in (">", "<", "==", "!=", ">=", "<=")):
                        in_filter = True
                        break
                    if bw in ("where", "between", "filter"):
                        in_filter = True
                        break
                if not in_filter:
                    prev_ends_comma = idx > 0 and raw_words[idx - 1][2].endswith(",")
                    next_word = (
                        raw_words[idx + 1][2].lower().strip(".,;:")
                        if idx + 1 < len(raw_words) else ""
                    )
                    if prev_ends_comma or next_word in target_verbs:
                        split_positions.add(s)
                        continue

            if w.endswith(",") and idx + 1 < len(raw_words):
                next_word = raw_words[idx + 1][2].lower().strip(".,;:")
                if next_word in target_verbs or next_word in ("then", "next", "afterwards"):
                    split_positions.add(e)
                    continue

        if not split_positions:
            return [text]

        sorted_splits = sorted(list(split_positions))
        clauses = []
        last_pos = 0
        for pos in sorted_splits:
            chunk = text[last_pos:pos].strip(" ,;:\n\t")
            if chunk.lower().startswith("and "):
                chunk = chunk[4:].strip(" ,;:\n\t")
            if chunk.lower().startswith("then "):
                chunk = chunk[5:].strip(" ,;:\n\t")
            if chunk:
                clauses.append(chunk)
            last_pos = pos

        tail = text[last_pos:].strip(" ,;:\n\t")
        if tail.lower().startswith("and "):
            tail = tail[4:].strip(" ,;:\n\t")
        if tail.lower().startswith("then "):
            tail = tail[5:].strip(" ,;:\n\t")
        if tail:
            clauses.append(tail)

        return clauses if clauses else [text]
