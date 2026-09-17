"""
src/route_methods/base.py - Neuro-Symbolic Topological Lattice (NSTL)
Abstract base class and shared utilities for pluggable RouteMethods (M0 - M6).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import math
import re
from typing import Dict, List, Optional, Set, Tuple, Any

from log_config import get_logger

try:
    from ..lattice import Cell, MicroCell, MacroCell, LatticeOrchestrator, TypeRegistry
    from ..unification import unify, Substitution, ExecutionContext
    from ..tokenizer import CellTokenizer
    from ..planner import LatticePlanner
except (ImportError, ValueError):
    from lattice import Cell, MicroCell, MacroCell, LatticeOrchestrator, TypeRegistry
    from unification import unify, Substitution, ExecutionContext
    from tokenizer import CellTokenizer
    from planner import LatticePlanner, STOPWORDS, _WILDCARD_CARRIERS

logger = get_logger("route_methods")

# STOPWORDS / _WILDCARD_CARRIERS are imported from planner (single source of
# truth — duplicated vocabularies silently drift and change token filtering
# between the router, the planner and the route methods).


class RouteMethod(ABC):
    """
    Abstract base class for all NSTL RouteMethods (M0 - M6).
    """
    name: str = "base"

    def __init__(self, orchestrator: Optional[LatticeOrchestrator] = None, **kwargs):
        self.orchestrator = orchestrator
        self.kwargs = kwargs

    @abstractmethod
    def plan(
        self,
        prompt: str,
        tunnel: List[Cell],
        relevance_map: Dict[str, float],
        orchestrator: Optional[LatticeOrchestrator] = None,
        ctx: Optional[ExecutionContext] = None,
        start_sig: Optional[Any] = None,
        goal_sig: Optional[Any] = None,
        max_transforms: int = 6,
        **kwargs
    ) -> List[Cell]:
        """
        Plans a type-valid sequence of Cell morphisms from the semantic tunnel.
        """
        pass

    def calculate_edge_affinity(self, src_cell: Cell, dst_cell: Cell, orchestrator: Optional[LatticeOrchestrator] = None) -> float:
        """
        Calculates empirical edge affinity score between two cells.
        AST-mined edges from real code snippets receive the highest affinity,
        followed by LLM seed edges, egress completion, and reachability.
        """
        orch = orchestrator or self.orchestrator

        # 1. Forward declared edges on src_cell
        dst_id_lower = dst_cell.cell_id.lower()
        for edge in getattr(src_cell, "edges", []):
            tgt_id = edge.get("target_cell_id") if isinstance(edge, dict) else getattr(edge, "target_cell_id", None)
            if tgt_id and (tgt_id == dst_cell.cell_id or str(tgt_id).lower() == dst_id_lower):
                aff = float(edge.get("affinity_score", 0.5) if isinstance(edge, dict) else getattr(edge, "affinity_score", 0.5))
                prov = edge.get("score_provenance") if isinstance(edge, dict) else getattr(edge, "score_provenance", "")
                if prov == "ast_mined":
                    return min(1.0, aff * 1.25)
                return aff

        # 2. Reverse declared edges on dst_cell
        src_id_lower = src_cell.cell_id.lower()
        for edge in getattr(dst_cell, "edges", []):
            tgt_id = edge.get("target_cell_id") if isinstance(edge, dict) else getattr(edge, "target_cell_id", None)
            if tgt_id and (tgt_id == src_cell.cell_id or str(tgt_id).lower() == src_id_lower):
                aff = float(edge.get("affinity_score", 0.3) if isinstance(edge, dict) else getattr(edge, "affinity_score", 0.3))
                return aff * 0.75

        # 3. Stage 2 -> Stage 3 Egress Completion
        src_stage = getattr(src_cell, "stage", None)
        dst_stage = getattr(dst_cell, "stage", None)
        src_domain = getattr(src_cell, "domain_name", "")
        dst_domain = getattr(dst_cell, "domain_name", "")
        if src_stage == 2 and dst_stage == 3 and src_domain and dst_domain and src_domain == dst_domain:
            return 0.75

        # 4. Topological reachability in orchestrator if built
        if orch:
            adj = getattr(orch, "_adjacency", None) or getattr(orch, "adjacency", None)
            if adj and dst_cell.cell_id in adj.get(src_cell.cell_id, ()):
                return 0.35

        # 5. Same-domain morphism continuity
        if src_domain and dst_domain and src_domain == dst_domain:
            return 0.20

        return 0.0

    def step_unifies(self, producer: Cell, consumer: Cell, prev_path: Optional[List[Cell]] = None) -> bool:
        """
        Checks if consumer cell can validly execute after producer cell.
        Enforces:
          1. Stage 1 cells cannot be appended as intermediate transitions.
          2. Type-monadic verification via LatticePlanner._verify_transition.
        """
        if getattr(consumer, "stage", None) == 1:
            return False

        orch = self.orchestrator
        if orch:
            planner = getattr(self, "_cached_planner", None)
            if planner is None:
                planner = LatticePlanner(orchestrator=orch)
                self._cached_planner = planner
            chain = prev_path if prev_path else [producer]
            sigma = planner._verify_transition(chain, consumer, Substitution())
            return sigma is not None

        # Fallback if orchestrator not provided
        req_inputs = [p for p in consumer.inputs.values() if p.required]
        prod_outputs = list(producer.outputs.values())
        if not prod_outputs:
            return False
        cand_inputs = req_inputs if req_inputs else list(consumer.inputs.values())
        for out_port in prod_outputs:
            for in_port in cand_inputs:
                if unify(out_port.signature, in_port.signature) is not None:
                    return True
        return False

    def find_bridge(
        self,
        src_cell: Cell,
        dst_cell: Cell,
        candidates: List[Cell],
        orchestrator: Optional[LatticeOrchestrator] = None
    ) -> Optional[Cell]:
        """
        Finds a 1-step bridging cell B in candidates such that:
        src_cell -> B unifies AND B -> dst_cell unifies.
        """
        best_b: Optional[Cell] = None
        best_score = -1.0

        for cand in candidates:
            if cand.cell_id in (src_cell.cell_id, dst_cell.cell_id):
                continue
            if getattr(cand, "stage", None) == 1:
                continue
            if self.step_unifies(src_cell, cand) and self.step_unifies(cand, dst_cell, prev_path=[src_cell, cand]):
                aff1 = self.calculate_edge_affinity(src_cell, cand, orchestrator)
                aff2 = self.calculate_edge_affinity(cand, dst_cell, orchestrator)
                score = aff1 + aff2
                if score > best_score:
                    best_score = score
                    best_b = cand

        return best_b

    def segment_prompt_clauses(self, prompt: str) -> List[str]:
        """
        Partitions user prompt into sequential procedural clauses.
        Delegates to the planner's single LANGUAGE-level segmenter so clause
        counts agree everywhere (the retired verb-lookahead splitter here used
        domain vocabulary as split triggers — the planner's segmenter splits
        only on punctuation/sequencing and merges list continuations).
        """
        try:
            from .planner import _segment_prompt_clauses
        except (ImportError, ValueError):
            from planner import _segment_prompt_clauses
        return _segment_prompt_clauses(prompt)
