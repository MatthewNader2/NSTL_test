"""
src/coverage_audit.py - Neuro-Symbolic Topological Lattice (NSTL)

Prompt-clause <-> pipeline-cell coverage audit. Corpus-derived (token_evidence),
vocabulary-free. Used by PreflightLinter (hard checks) and the CLI --debug panel
(so the person can SEE why a clause is unserved or a cell is unrequested).

Definitions
  winner(clause)   best-ranked on-path cell by (recall, name_precision, F) with identity evidence
  served(clause)   the cells ASSIGNED to the clause jointly explain as much of it (identity
                   token mass) as the best single cell anywhere in the lattice could
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

    # 1. evidence winner of every clause among the path's cells
    winners: Dict[int, int] = {}
    for j in range(len(clauses)):
        best_i, best_rank = None, None
        for i in range(len(cells)):
            o = table[i][j]
            if o and o["hit"] > 0.0 and (best_rank is None or o["rank"] > best_rank):
                best_i, best_rank = i, o["rank"]
        if best_i is not None:
            winners[j] = best_i

    # 2. which clauses does each cell CLAIM? An explicit assignment (planner / tagger) is authoritative;
    #    otherwise a cell claims the clauses it wins plus those where it explains tokens the winner left over.
    bridge_roles = set()
    try:
        from lattice import TypeRegistry
        bridge_roles = {str(r).lower() for r in TypeRegistry.get_instance().get_verification_semantics("bridge_roles")}
    except Exception:
        pass
    claims: List[set] = []
    cell_rows: List[Dict[str, Any]] = []
    for i, cell in enumerate(cells):
        touched = [j for j in range(len(clauses)) if table[i][j] and table[i][j]["hit"] > 0.0]
        won = [j for j in touched if winners.get(j) == i]
        residual_serves = []
        for j in touched:
            if j in won:
                continue
            w = table[winners[j]][j]
            if ev.identity_overlap(ctoks[j] - set(w["tokens"]), cell)["hit"] > 0.0:
                residual_serves.append(j)
        tagged = getattr(cell, "matched_clause_idx", None)
        mine = {tagged} if tagged is not None else (set(won) | set(residual_serves))
        claims.append(mine)
        role = str(getattr(cell, "node_role", "") or "").lower()
        if mine:
            status = "serves"
        elif not touched or role in bridge_roles:
            status = "connector"
        else:
            status = "UNREQUESTED"
        cell_rows.append({"cell": cell.cell_id, "wins": sorted(set(won) | ({tagged} if tagged is not None else set())),
                          "residual_serves": residual_serves, "touches": touched, "status": status})
        if status == "UNREQUESTED":
            out["issues"].append({"kind": "unrequested_cell", "cell": cell.cell_id, "outranked_on": touched,
                                  "winners": {j: cells[winners[j]].cell_id for j in touched if j in winners}})
    out["cells"] = cell_rows

    # 3. a clause is SERVED only by the cells assigned to it, jointly explaining as much of it as the
    #    best single lattice cell could (incidental hits by other cells no longer count: an optimiser
    #    exploits a whole-path union immediately).
    for j, toks in enumerate(ctoks):
        claimed = [i for i in range(len(cells)) if j in claims[i] and table[i][j]]
        lattice_best = lattice_best_fit = 0.0
        if toks:
            memo = ev.__dict__.setdefault("_lattice_best_memo", {})
            key = frozenset(toks)
            if key not in memo:
                lb = lbf = 0.0
                for lc in getattr(ev.orch, "loaded_cells", {}).values():
                    lo = ev.identity_overlap(toks, lc)
                    # fit = how much of the clause the cell explains x how completely the clause NAMES the cell
                    lb, lbf = max(lb, lo["recall"]), max(lbf, lo["recall"] * lo["name_precision"])
                memo[key] = (lb, lbf)
            lattice_best, lattice_best_fit = memo[key]
        c_mass = ev.mass(toks)
        path_tokens: set = set()
        for i in claimed:
            path_tokens |= set(table[i][j]["tokens"])
        path_best = (ev.mass(path_tokens) / c_mass) if c_mass > 0 else 0.0
        quality = (max(table[i][j]["recall"] * table[i][j]["name_precision"] for i in claimed) / lattice_best_fit) if (claimed and lattice_best_fit > 0) else 0.0
        if not toks or lattice_best <= 0.0:
            status = "no-evidence" if not toks else "unservable"
        else:
            status = "served" if path_best >= lattice_best - 1e-9 else "UNSERVED"
        best_i = winners.get(j)
        out["clauses"].append({
            "idx": j, "text": clauses[j], "tokens": sorted(toks),
            "winner": cells[best_i].cell_id if best_i is not None else None,
            "claimed_by": [cells[i].cell_id for i in claimed],
            "path_recall": round(path_best, 3), "lattice_best_recall": round(lattice_best, 3),
            "quality": round(min(quality, 1.0), 3), "status": status})
        if status == "UNSERVED":
            out["issues"].append({"kind": "clause_unserved", "clause_idx": j, "clause": clauses[j],
                                  "path_recall": round(path_best, 3), "lattice_best_recall": round(lattice_best, 3),
                                  "best_on_path": cells[best_i].cell_id if best_i is not None else None})
    return out
