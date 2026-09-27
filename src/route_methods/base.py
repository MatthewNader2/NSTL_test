"""
src/route_methods/base.py - Neuro-Symbolic Topological Lattice (NSTL)
Abstract base class and shared utilities for pluggable RouteMethods (M0 - M6).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import math
from typing import Dict, List, Optional, Set, Tuple, Any

from log_config import get_logger

try:
    from ..lattice import Cell, MicroCell, MacroCell, LatticeOrchestrator, TypeRegistry, is_path_port
    from ..unification import unify, Substitution, ExecutionContext, unify_cell_with_scope
    from ..tokenizer import CellTokenizer
    from ..planner import LatticePlanner
except (ImportError, ValueError):
    from lattice import Cell, MicroCell, MacroCell, LatticeOrchestrator, TypeRegistry, is_path_port
    from unification import unify, Substitution, ExecutionContext, unify_cell_with_scope
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

    def calculate_edge_affinity(
        self,
        src_cell: Cell,
        dst_cell: Cell,
        orchestrator: Optional[LatticeOrchestrator] = None,
        relevance_map: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculates empirical edge affinity score between two cells.
        AST-mined edges from real code snippets receive the highest affinity,
        followed by LLM seed edges, egress completion, and reachability.
        """
        orch = orchestrator or self.orchestrator

        # Synaptic Macro-Goal Edge Reinforcement (Synapses in the Brain):
        try:
            from config import settings
            macros_enabled = bool(getattr(settings, "macros_enabled", True))
        except Exception:
            macros_enabled = True

        if macros_enabled and orch and hasattr(orch, "loaded_cells"):
            for m in orch.loaded_cells.values():
                if getattr(m, "cell_type", "") == "macro" or getattr(m, "sub_cells", None):
                    topo = getattr(m, "internal_topology", {}) or {}
                    is_macro_edge = dst_cell.cell_id in topo.get(src_cell.cell_id, ())
                    if not is_macro_edge and getattr(m, "sub_cells", None):
                        subs = m.sub_cells
                        for i in range(len(subs) - 1):
                            if subs[i] == src_cell.cell_id and subs[i + 1] == dst_cell.cell_id:
                                is_macro_edge = True
                                break
                    if is_macro_edge:
                        return 0.85

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
            # A.2 Fix: Gate the +0.75 egress completion bonus on src_cell having positive prompt relevance
            rel_map = relevance_map if relevance_map is not None else getattr(self, "relevance_map", None)
            if rel_map is None or rel_map.get(src_cell.cell_id, 0.0) > 0.0:
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

    def _get_prompt_literals(self, prompt: Optional[str] = None) -> Tuple[List[Any], List[Any], List[Any]]:
        p = prompt or getattr(self, "_current_prompt", "") or ""
        if not p:
            return [], [], []
        cached = getattr(self, "_cached_literals", None)
        if cached is not None and cached[0] == p:
            return cached[1]
        universal = ExecutionContext._extract_universal_literals(p)
        id_lits = [v for _, k, v in universal if k == "identifier"]
        qs_lits = [v for _, k, v in universal if k == "quoted_str"]
        num_lits = [v for _, k, v in universal if k == "numeric"]
        res = (id_lits, qs_lits, num_lits)
        self._cached_literals = (p, res)
        return res

    def step_unifies(self, producer: Cell, consumer: Cell, prev_path: Optional[List[Cell]] = None, prompt: Optional[str] = None) -> bool:
        """
        Checks if consumer cell can validly execute after producer cell.
        Enforces:
          1. Stage 1 cells cannot be appended as intermediate transitions.
          2. Entity-grounded transition check: producer and consumer must not operate on
             disjoint prompt entities unless consumer is a multi-input join morphism.
          3. Type-monadic verification via LatticePlanner._verify_transition with prompt literals.
        """
        if getattr(consumer, "stage", None) == 1:
            return False

        # Entity-grounded transition check:
        prod_lits = set(getattr(producer, "clause_literals", None) or ())
        cons_lits = set(getattr(consumer, "clause_literals", None) or ())
        if prod_lits and cons_lits and not (prod_lits & cons_lits):
            is_join = len([p for p in consumer.inputs.values() if p.required]) >= 2
            if not is_join:
                return False

        orch = self.orchestrator
        chain = prev_path if prev_path else [producer]
        if orch:
            planner = getattr(self, "_cached_planner", None)
            if planner is None:
                planner = LatticePlanner(orchestrator=orch)
                self._cached_planner = planner
            id_lits, qs_lits, num_lits = self._get_prompt_literals(prompt)
            sigma = planner._verify_transition(
                chain,
                consumer,
                Substitution(),
                identifier_literals=id_lits,
                quoted_str_literals=qs_lits,
                numeric_literals=num_lits,
            )
            if sigma is not None:
                return True
            # Multi-carrier DAG scope verification:
            dag_res = unify_cell_with_scope(consumer, chain, Substitution())
            return dag_res is not None

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
        if prev_path and len(prev_path) > 1:
            for anc in reversed(prev_path[:-1]):
                for out_port in anc.outputs.values():
                    for in_port in cand_inputs:
                        if unify(out_port.signature, in_port.signature) is not None:
                            return True
        dag_res = unify_cell_with_scope(consumer, chain, Substitution())
        return dag_res is not None

    def step_unifies_dag(
        self,
        consumer: Cell,
        available_cells: List[Cell],
        sigma: Optional[Substitution] = None,
        ctx: Optional[ExecutionContext] = None,
    ) -> bool:
        """
        DAG-frontier unification check: verifies if consumer can validly execute
        given the full multi-carrier scope of available ancestor cells (fork/join).
        """
        if getattr(consumer, "stage", None) == 1 or not available_cells:
            return False
        res = unify_cell_with_scope(consumer, available_cells, sigma=sigma, context=ctx)
        return res is not None

    def find_bridge(
        self,
        src_cell: Cell,
        dst_cell: Cell,
        candidates: List[Cell],
        orchestrator: Optional[LatticeOrchestrator] = None,
        prev_path: Optional[List[Cell]] = None
    ) -> Optional[Cell]:
        """
        Finds a 1-step bridging cell B in candidates such that:
        B unifies with the current scope/src_cell AND dst_cell unifies after B.
        Returns a cloned instance parameter-grounded to dst_cell's clause literals.
        """
        best_b: Optional[Cell] = None
        best_score = -1.0
        current_scope = prev_path if prev_path else [src_cell]

        for cand in candidates:
            if cand.cell_id in (src_cell.cell_id, dst_cell.cell_id):
                continue
            if getattr(cand, "stage", None) == 1:
                continue
            if getattr(cand, "role", "") == "combinator" or getattr(cand, "node_type", "") == "combinator":
                continue
            if self.step_unifies(src_cell, cand, prev_path=current_scope) and self.step_unifies(cand, dst_cell, prev_path=current_scope + [cand]):
                aff1 = self.calculate_edge_affinity(src_cell, cand, orchestrator)
                aff2 = self.calculate_edge_affinity(cand, dst_cell, orchestrator)
                score = aff1 + aff2
                if score > best_score:
                    best_score = score
                    best_b = cand

        if best_b:
            b_inst = best_b.clone() if hasattr(best_b, "clone") else best_b
            b_inst.clause_literals = getattr(dst_cell, "clause_literals", None)
            b_inst.matched_clause_idx = getattr(dst_cell, "matched_clause_idx", None)
            return b_inst
        return None

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

    def tag_cells_with_clause_indices(self, path: List[Cell], prompt: str) -> None:
        """
        Tags each cell in path with matched_clause_idx based on prompt clauses.
        Preserves intentional sub-goals (e.g. mean, normalize) during dead-code pruning
        and provides clause-scoped literal binding.
        """
        if not path or not prompt:
            return
        from lattice import CellTokenizer
        clauses = self.segment_prompt_clauses(prompt)
        if not clauses:
            return
        clause_toks = [CellTokenizer.tokenize_prompt(cl) for cl in clauses]
        for cell in path:
            if getattr(cell, "matched_clause_idx", None) is not None:
                continue
            c_toks = getattr(cell, "token_set", set())
            id_toks = getattr(cell, "identity_tokens", c_toks)
            best_idx = None
            best_score = 0.0
            for idx, cl_tok in enumerate(clause_toks):
                if not cl_tok:
                    continue
                strong = len(cl_tok & id_toks)
                weak = len((cl_tok & c_toks) - id_toks)
                score = (strong * 3.0) + weak
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_score > 0.0:
                cell.matched_clause_idx = best_idx

    def select_stratified_candidates(
        self,
        prompt: str,
        candidates: List[Cell],
        relevance_map: Dict[str, float],
        orchestrator: Optional[LatticeOrchestrator] = None,
        max_shown: int = 40,
    ) -> List[Cell]:
        """
        Chronological clause-stratified candidate selection:
        Guarantees that every procedural clause of a compound prompt contributes its
        most precise matching operations, plus carrier bridges, preventing clause starvation
        and keeping the candidate list clean and focused for local LLM reasoning.
        """
        self._current_prompt = prompt
        orch = orchestrator or self.orchestrator
        clauses = self.segment_prompt_clauses(prompt)
        prompt_lower = prompt.lower()

        selected_cells: List[Cell] = []
        selected_ids: Set[str] = set()

        filtered_cands: List[Cell] = []
        for c in candidates:
            cid = c.cell_id
            if getattr(c, "node_type", "") == "constant" or cid.startswith("MACRO_"):
                continue
            if getattr(c, "role", "") == "combinator" or getattr(c, "node_type", "") == "combinator":
                continue
            filtered_cands.append(c)

        for idx, cl in enumerate(clauses):
            cl_toks = CellTokenizer.tokenize_prompt(cl) - STOPWORDS
            if not cl_toks:
                continue

            if idx == 0 and len(clauses) > 1:
                stage_pool = [c for c in filtered_cands if getattr(c, "stage", None) == 1] or filtered_cands
            else:
                stage_pool = [c for c in filtered_cands if getattr(c, "stage", None) != 1] or filtered_cands

            scored: List[Tuple[Cell, float]] = []
            for c in stage_pool:
                cid = c.cell_id
                cid_lower = cid.lower()
                c_toks = c.token_set - STOPWORDS
                id_toks = getattr(c, "identity_tokens", c_toks) - STOPWORDS

                strong = len(cl_toks & id_toks)
                if strong == 0 and relevance_map.get(cid, 0.0) < 0.15:
                    continue
                weak = len((cl_toks & c_toks) - id_toks)

                cid_toks = CellTokenizer.tokenize_identifier(cid_lower)
                id_match = sum(2.0 for t in cl_toks if t in cid_toks or t in c.keywords)
                rel = relevance_map.get(cid, 0.0)
                score = (strong * 4.0) + (weak * 1.5) + id_match + (rel * 2.0)
                scored.append((c, score))

            if not scored:
                continue
            scored.sort(key=lambda x: x[1], reverse=True)
            top_sc = scored[0][1]
            for c, sc in scored[:2]:
                if sc < top_sc * 0.70:
                    break
                if c.cell_id not in selected_ids:
                    selected_cells.append(c)
                    selected_ids.add(c.cell_id)

        # Carrier bridges: active domain bridges via ontology metadata
        active_domains = {getattr(c, "domain_name", "") for c in selected_cells if getattr(c, "domain_name", "")}
        all_cells = list(orch.loaded_cells.values()) if orch and hasattr(orch, "loaded_cells") else candidates
        for c in all_cells:
            cid = c.cell_id
            c_dom = getattr(c, "domain_name", "")
            if c_dom and c_dom not in active_domains and not any(d in cid.lower() for d in active_domains):
                continue
            if getattr(c, "role", "") == "combinator":
                continue
            is_bridge = (
                getattr(c, "node_role", "") == "bridge"
                or getattr(c, "node_type", "") == "tunnel"
            )
            if is_bridge and cid not in selected_ids:
                selected_cells.append(c)
                selected_ids.add(cid)

        # Top globally relevant cells fill up only if sparse
        if len(selected_cells) < 12:
            sorted_by_rel = sorted(filtered_cands, key=lambda c: relevance_map.get(c.cell_id, 0.0), reverse=True)
            for c in sorted_by_rel:
                if len(selected_cells) >= max_shown or len(selected_cells) >= 16:
                    break
                if c.cell_id not in selected_ids:
                    selected_cells.append(c)
                    selected_ids.add(c.cell_id)

        return selected_cells

