"""
src/coverage_audit.py - Neuro-Symbolic Topological Lattice (NSTL)

Prompt-clause <-> pipeline-cell coverage audit. Corpus-derived (token_evidence),
vocabulary-free. Used by PreflightLinter (hard checks) and the CLI --debug panel
(so the person can SEE why a clause is unserved or a cell is unrequested).

Definitions
  winner(clause)   best-ranked on-path cell by (recall, name_precision, F) with identity evidence
  served(clause)   the winner explains as much of the clause (identity recall) as the best
                   cell anywhere in the lattice could (ties count)
  unrequested(cell) has identity evidence somewhere, is outranked on every clause it
                   touches, and explains no residual clause token the winner left over.
                   Cells with zero evidence anywhere are connectors (projection/cast): never flagged.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    from .token_evidence import get_evidence, last_orchestrator
except (ImportError, ValueError):
    from token_evidence import get_evidence, last_orchestrator


def _clauses(prompt: str) -> List[str]:
    try:
        from route_methods.base import RouteMethod
    except ImportError:  # package-style import
        from .route_methods.base import RouteMethod
    return RouteMethod.segment_prompt_clauses(prompt)


def audit(cells: List[Any], prompt: str, orchestrator: Any = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False, "clauses": [], "cells": [], "issues": []}
    ev = get_evidence(orchestrator or last_orchestrator())
    if ev is None or not prompt or not cells:
        return out
    clauses = _clauses(prompt)
    ctoks = [ev.clause_tokens(c) for c in clauses]
    table = [[ev.identity_overlap(t, c) if t else None for t in ctoks] for c in cells]
    out["available"] = True
    winners: Dict[int, int] = {}
    for j, toks in enumerate(ctoks):
        best_i, best_rank = None, None
        for i in range(len(cells)):
            o = table[i][j]
            if o and o["hit"] > 0.0 and (best_rank is None or o["rank"] > best_rank):
                best_i, best_rank = i, o["rank"]
        if best_i is not None:
            winners[j] = best_i
        w = table[best_i][j] if best_i is not None else None
        # served == the on-path winner explains as much of the clause as the BEST cell
        # anywhere in the lattice could (ties count). Parameter-free, vocabulary-free.
        lattice_best = max((ev.identity_overlap(toks, lc)["recall"]
                            for lc in getattr(ev.orch, "loaded_cells", {}).values()), default=0.0) if toks else 0.0
        path_best = w["recall"] if w else 0.0
        if not toks or lattice_best <= 0.0:
            status = "no-evidence" if not toks else "unservable"
        else:
            status = "served" if path_best >= lattice_best - 1e-9 else "UNSERVED"
        out["clauses"].append({
            "idx": j, "text": clauses[j], "tokens": sorted(toks),
            "winner": cells[best_i].cell_id if best_i is not None else None,
            "path_recall": round(path_best, 3), "lattice_best_recall": round(lattice_best, 3),
            "status": status})
        if status == "UNSERVED":
            out["issues"].append({"kind": "clause_unserved", "clause_idx": j, "clause": clauses[j],
                                  "path_recall": round(path_best, 3), "lattice_best_recall": round(lattice_best, 3),
                                  "best_on_path": cells[best_i].cell_id if best_i is not None else None})
    bridge_roles = set()
    try:
        from lattice import TypeRegistry
        bridge_roles = {str(r).lower() for r in TypeRegistry.get_instance().get_verification_semantics("bridge_roles")}
    except Exception:
        pass
    for i, cell in enumerate(cells):
        touched = [j for j in range(len(clauses)) if table[i][j] and table[i][j]["hit"] > 0.0]
        won = [j for j in touched if winners.get(j) == i]
        residual_serves = []
        for j in touched:
            if j in won:
                continue
            w = table[winners[j]][j]
            residual = ctoks[j] - set(w["tokens"])
            if ev.identity_overlap(residual, cell)["hit"] > 0.0:
                residual_serves.append(j)
        role = str(getattr(cell, "node_role", "") or "").lower()
        if not touched:
            status = "connector"
        elif won or residual_serves:
            status = "serves"
        elif role in bridge_roles:
            status = "connector"
        else:
            status = "UNREQUESTED"
        out["cells"].append({"cell": cell.cell_id, "wins": won, "residual_serves": residual_serves,
                             "touches": touched, "status": status})
        if status == "UNREQUESTED":
            out["issues"].append({"kind": "unrequested_cell", "cell": cell.cell_id, "outranked_on": touched,
                                  "winners": {j: cells[winners[j]].cell_id for j in touched}})
    return out
