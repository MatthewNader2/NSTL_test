"""
src/token_evidence.py - Neuro-Symbolic Topological Lattice (NSTL)

Single, corpus-derived source of lexical evidence. Every consumer that asks
"how strongly does this clause point at this cell?" (router, route methods,
clause tagging, preflight coverage audit) goes through here, so the answer is
identical everywhere and contains NO vocabulary: all weights come from the
loaded lattice's own token index (inverse document frequency over cells).

Principles
  * A token that no loaded cell can ever match carries zero evidence (it is
    filler for THIS lattice: "called", "then", "it", ...). No stoplist needed.
  * A token's weight is its IDF over the cell corpus.
  * A cell is judged by IDENTITY evidence only (operation id + declared
    keywords). Docstring/port vocabulary is weak evidence and never claims a
    clause on its own.
  * Precision matters as much as recall: a cell whose identity is mostly
    NOT mentioned by the clause (get_dummies for "get ...") is a poor claim.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Set

try:
    from .tokenizer import CellTokenizer
except (ImportError, ValueError):
    from tokenizer import CellTokenizer


class TokenEvidence:
    def __init__(self, orchestrator: Any):
        self.orch = orchestrator
        cells = list(getattr(orchestrator, "loaded_cells", {}).values())
        self.n = max(len(cells), 1)
        df: Dict[str, int] = {}
        for c in cells:
            for t in c.token_set:
                df[t] = df.get(t, 0) + 1
        self._df = df
        self._idf_cache: Dict[str, float] = {}

    def idf(self, tok: str) -> float:
        """0.0 for tokens no cell can match; log-IDF otherwise."""
        v = self._idf_cache.get(tok)
        if v is None:
            d = self._df.get(tok, 0)
            v = 0.0 if d == 0 else math.log(1.0 + (self.n + 1) / (d + 1.0))
            self._idf_cache[tok] = v
        return v

    def mass(self, toks: Iterable[str]) -> float:
        return sum(self.idf(t) for t in toks)

    def clause_tokens(self, clause: str) -> Set[str]:
        return {t for t in CellTokenizer.tokenize_prompt(clause) if self.idf(t) > 0.0}

    def identity_overlap(self, clause_toks: Set[str], cell: Any) -> Dict[str, float]:
        """Evidence a clause gives for a cell.
        recall    = share of the clause's mass explained by the cell identity
        precision = share of the cell identity's mass mentioned by the clause
        """
        ident = set(getattr(cell, "identity_tokens", None) or set())
        hit = clause_toks & ident
        hit_mass = self.mass(hit)
        c_mass = self.mass(clause_toks)
        i_mass = self.mass(ident)
        recall = hit_mass / c_mass if c_mass > 0 else 0.0
        precision = hit_mass / i_mass if i_mass > 0 else 0.0
        denom = precision + recall
        f = (2 * precision * recall / denom) if denom > 0 else 0.0
        # How completely the clause NAMES the operation (cell id tokens only).
        # Distinguishes fft from ifftshift when both share the same keywords:
        # the clause says "fft", the id of ifftshift also says "shift".
        name_toks = {t for t in CellTokenizer.tokenize_identifier(str(getattr(cell, "cell_id", ""))) if self.idf(t) > 0.0}
        n_mass = self.mass(name_toks)
        name_precision = (self.mass(clause_toks & name_toks) / n_mass) if n_mass > 0 else 0.0
        return {"hit": hit_mass, "recall": recall, "precision": precision, "f": f,
                "name_precision": name_precision,
                "rank": (recall, name_precision, f),
                "tokens": sorted(hit)}


_CACHE: Dict[int, "TokenEvidence"] = {}
_LAST: list = []


def last_orchestrator() -> Any:
    """Most recent orchestrator an evidence service was built for (lint has no handle on it)."""
    return _LAST[0] if _LAST else None


def get_evidence(orchestrator: Any) -> Optional[TokenEvidence]:
    """Per-orchestrator singleton, rebuilt if the loaded cell count changes."""
    if orchestrator is None or not getattr(orchestrator, "loaded_cells", None):
        return None
    key = id(orchestrator)
    ev = _CACHE.get(key)
    if ev is None or ev.n != max(len(orchestrator.loaded_cells), 1):
        ev = TokenEvidence(orchestrator)
        _CACHE[key] = ev
    _LAST[:] = [orchestrator]
    return ev
