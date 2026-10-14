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
    from ..utils import tokenize_alphanumeric
    from ..dag_plan import DAGPlan, OpenPortFrontier
except (ImportError, ValueError):
    from lattice import Cell, MicroCell, MacroCell, LatticeOrchestrator, TypeRegistry, is_path_port
    from unification import unify, Substitution, ExecutionContext, unify_cell_with_scope
    from tokenizer import CellTokenizer
    from planner import LatticePlanner, STOPWORDS, _WILDCARD_CARRIERS
    from utils import tokenize_alphanumeric
    from dag_plan import DAGPlan, OpenPortFrontier

logger = get_logger("route_methods")

# STOPWORDS / _WILDCARD_CARRIERS are imported from planner (single source of
# truth — duplicated vocabularies silently drift and change token filtering
# between the router, the planner and the route methods).


_ACTIVE_ORCHESTRATOR: List[Any] = []


def _bridge_roles() -> frozenset:
    """Declared bridge/adapter role names come from the trees' verification_semantics."""
    try:
        from lattice import TypeRegistry
        return frozenset(str(r).strip().lower() for r in TypeRegistry.get_instance().get_verification_semantics("bridge_roles"))
    except Exception:
        return frozenset()


def is_terminal_sink(cell: Any, orch: Optional[Any] = None) -> bool:
    """Check if cell is a terminal/sink morphism dynamically via LatticeVocabulary and TypeRegistry."""
    try:
        from planner import _active_vocab
        return _active_vocab(orch).is_terminal(cell)
    except Exception:
        pass
    try:
        from lattice import TypeRegistry
        reg = TypeRegistry.get_instance()
        term_roles = {str(r).lower() for r in reg.get_verification_semantics("terminal_roles")}
        term_roles.update(str(r).lower() for r in reg.get_verification_semantics("sink_roles"))
        role = str(getattr(cell, "node_role", "") or getattr(cell, "role", "") or "").lower()
        if role in term_roles:
            return True
        term_stages = reg.get_verification_semantics("terminal_stages")
        if term_stages and getattr(cell, "stage", None) in term_stages:
            return True
    except Exception:
        pass
    if getattr(cell, "is_endable", False):
        return True
    stage = getattr(cell, "stage", None)
    role = str(getattr(cell, "node_role", "") or getattr(cell, "role", "") or "").lower()
    return stage == 3 or role in ("sink", "terminal", "egress", "consumer")


def is_ingress_source(cell: Any, orch: Optional[Any] = None) -> bool:
    """Check if cell is an ingress/source morphism dynamically via LatticeVocabulary and TypeRegistry."""
    try:
        from planner import _active_vocab
        return _active_vocab(orch).is_ingress(cell)
    except Exception:
        pass
    try:
        from lattice import TypeRegistry
        reg = TypeRegistry.get_instance()
        ing_roles = {str(r).lower() for r in reg.get_verification_semantics("ingress_roles")}
        ing_roles.update(str(r).lower() for r in reg.get_verification_semantics("source_roles"))
        role = str(getattr(cell, "node_role", "") or getattr(cell, "role", "") or "").lower()
        if role in ing_roles:
            return True
        ing_stages = reg.get_verification_semantics("ingress_stages")
        if ing_stages and getattr(cell, "stage", None) in ing_stages:
            return True
    except Exception:
        pass
    stage = getattr(cell, "stage", None)
    role = str(getattr(cell, "node_role", "") or getattr(cell, "role", "") or "").lower()
    return stage == 1 or role in ("source", "ingress", "producer")


def is_intermediate_transform(cell: Any, orch: Optional[Any] = None) -> bool:
    """Check if cell is an intermediate/transform morphism dynamically via LatticeVocabulary and TypeRegistry."""
    try:
        from planner import _active_vocab
        return _active_vocab(orch).is_transform(cell)
    except Exception:
        pass
    try:
        from lattice import TypeRegistry
        reg = TypeRegistry.get_instance()
        tf_roles = {str(r).lower() for r in reg.get_verification_semantics("transform_roles")}
        role = str(getattr(cell, "node_role", "") or getattr(cell, "role", "") or "").lower()
        if role in tf_roles:
            return True
        tf_stages = reg.get_verification_semantics("transform_stages")
        if tf_stages and getattr(cell, "stage", None) in tf_stages:
            return True
    except Exception:
        pass
    stage = getattr(cell, "stage", None)
    role = str(getattr(cell, "node_role", "") or getattr(cell, "role", "") or "").lower()
    return stage == 2 or role in ("transform", "operation", "intermediate")


class RouteMethod(ABC):
    """
    Abstract base class for all NSTL RouteMethods (M0 - M6).
    """
    name: str = "base"

    def __init__(self, orchestrator: Optional[LatticeOrchestrator] = None, **kwargs):
        self.orchestrator = orchestrator
        self.kwargs = kwargs
        self.trace_events: List[Dict[str, Any]] = []

    effective_method: Optional[str] = None      # set when this method handed the request to another
    fallback_reason: Optional[str] = None

    def _delegate_to_m1(self, reason: str, prompt: str, tunnel: List[Cell], relevance_map: Dict[str, float],
                        orch: Any = None, **plan_kwargs: Any) -> List[Cell]:
        """
        Hand the request to M1 (clause anchor) AND SAY SO. M4/M5/M7/M8/M9 used to call M1 silently
        (no model, bad JSON, unbridgeable step, short chain, ...), so a row labelled M4 in a
        benchmark could be an M1 result with nothing in the log to show it.
        """
        from route_methods.m1_clause_anchor import M1ClauseAnchorRouteMethod
        self.effective_method = "M1"
        self.fallback_reason = reason
        self._trace("method_fallback", requested=type(self).__name__, to="M1", reason=reason)
        m1 = M1ClauseAnchorRouteMethod(orchestrator=orch or self.orchestrator)
        out = m1.plan(prompt, tunnel, relevance_map, orchestrator=orch or self.orchestrator, **plan_kwargs)
        self.trace_events.extend(getattr(m1, "trace_events", []))
        return out

    def _trace(self, event: str, **fields: Any) -> None:
        """Structured planner telemetry surfaced by `--debug` (never affects planning)."""
        if not hasattr(self, "trace_events"):
            self.trace_events = []
        self.trace_events.append({"event": event, **fields})

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

        # 3. Intermediate -> Terminal Sink Egress Completion
        src_domain = getattr(src_cell, "domain_name", "")
        dst_domain = getattr(dst_cell, "domain_name", "")
        is_src_intermediate = is_intermediate_transform(src_cell, orch)
        is_dst_sink = is_terminal_sink(dst_cell, orch)
        if is_src_intermediate and is_dst_sink and src_domain and dst_domain and src_domain == dst_domain:
            # Gate the +0.75 egress completion bonus on src_cell having positive prompt relevance
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

    @staticmethod
    def is_file_format_compatible(cell: Cell, file_path_str: str) -> bool:
        """
        Checks if a cell's file/path port is compatible with the given file path's extension.
        100% domain-agnostic: derives declared format from generic signature File[Carrier, Format]
        or declared cell metadata.
        """
        if not file_path_str or not isinstance(file_path_str, str) or "." not in file_path_str:
            return True
        ext = file_path_str.rpartition(".")[2].strip().lower()
        if not ext:
            return True

        path_ports = [
            p for p in list(cell.inputs.values()) + list(cell.outputs.values())
            if is_path_port(p)
        ]
        if not path_ports:
            return True

        for p in path_ports:
            tn = str(getattr(p, "type_name", "") or "").strip()
            if "[" in tn and tn.endswith("]"):
                ctor, _, inner = tn[:-1].partition("[")
                if ctor.strip().lower() in ("file", "stream", "path", "asset"):
                    parts = [part.strip().lower() for part in inner.split(",") if part.strip()]
                    if len(parts) >= 2:
                        fmt = parts[1]
                        reg = TypeRegistry.get_instance()
                        if fmt in ("any", "generic", "*", ""):
                            carrier = parts[0]
                            try:
                                placeholder = reg.get_asset_placeholder(carrier, "")
                                if placeholder and "." in placeholder:
                                    carrier_ext = placeholder.rpartition(".")[2].strip().lower()
                                    if carrier_ext and carrier_ext != "dat":
                                        if ext == carrier_ext or reg.is_subtype(ext, carrier_ext) or reg.is_subtype(carrier_ext, ext):
                                            return True
                                        for anc in reg.get_format_groups():
                                            if reg.is_subtype(ext, anc) and reg.is_subtype(carrier_ext, anc):
                                                return True
                                        return False
                            except Exception:
                                pass
                            return True
                        if ext == fmt or fmt in ext or ext in fmt or reg.is_subtype(ext, fmt) or reg.is_subtype(fmt, ext):
                            return True
                        for anc in reg.get_format_groups():
                            if reg.is_subtype(ext, anc) and (reg.is_subtype(fmt, anc) or fmt in anc):
                                return True
                        return False
        return True

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
        Checks if consumer cell can validly execute in the multi-carrier environment.
        Enforces:
          1. Stage 1 cells cannot be appended as intermediate transitions.
          2. Type-monadic verification via LatticePlanner._verify_transition with prompt literals.
          3. Multi-carrier DAG scope verification via unify_cell_with_scope across available path wires.
        """
        if getattr(consumer, "stage", None) == 1:
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

        prompt_str = getattr(self, "prompt", "") or (getattr(orchestrator, "prompt", "") if orchestrator else "")
        prompt_l = str(prompt_str).lower()

        for cand in candidates:
            if cand.cell_id in (src_cell.cell_id, dst_cell.cell_id):
                continue
            if is_ingress_source(cand, orchestrator):
                continue
            if getattr(cand, "is_combinator", False) or getattr(cand, "node_role", "") == "combinator" or getattr(cand, "role", "") == "combinator" or getattr(cand, "node_type", "") == "combinator":
                continue
            if is_terminal_sink(cand, orchestrator):
                continue

            cand_role = str(getattr(cand, "node_role", "") or "").lower()
            is_declared_bridge = cand_role in _bridge_roles()
            resolves_property_mismatch = False

            # Check if cand resolves a property precondition of dst_cell violated by src_cell / current_scope
            dst_pres = getattr(dst_cell, "preconditions", None) or []
            src_posts = getattr(src_cell, "postconditions", None) or []
            cand_posts = getattr(cand, "postconditions", None) or []

            def _c_prop(c, key, default=None):
                if isinstance(c, dict):
                    return c.get(key, default)
                return getattr(c, key, default)

            if dst_pres and cand_posts:
                for dp in dst_pres:
                    dp_prop = _c_prop(dp, "property")
                    dp_val = _c_prop(dp, "value")
                    dp_op = _c_prop(dp, "operator")
                    if dp_op == "==" and dp_prop is not None:
                        src_conflict = any(
                            _c_prop(sp, "property") == dp_prop
                            and _c_prop(sp, "operator") == "=="
                            and _c_prop(sp, "value") != dp_val
                            for sp in src_posts
                        )
                        cand_resolves = any(
                            _c_prop(cp, "property") == dp_prop
                            and _c_prop(cp, "operator") == "=="
                            and _c_prop(cp, "value") == dp_val
                            for cp in cand_posts
                        )
                        if src_conflict and cand_resolves:
                            resolves_property_mismatch = True
                            break

            if not is_declared_bridge and not resolves_property_mismatch:
                # Operational transforms require positive lexical/semantic evidence in the prompt to act as bridges
                cand_toks = set(getattr(cand, "token_set", set())) | set(getattr(cand, "keywords", []) or []) | set(getattr(cand, "semantic_tags", []) or [])
                cand_toks = {str(t).lower() for t in cand_toks}
                prompt_toks = set(tokenize_alphanumeric(prompt_l))
                if not (cand_toks & prompt_toks):
                    continue
            can_src_to_cand = self.step_unifies(src_cell, cand, prev_path=current_scope) or self.step_unifies_dag(cand, current_scope)
            can_cand_to_dst = self.step_unifies(cand, dst_cell, prev_path=current_scope + [cand]) or self.step_unifies_dag(dst_cell, current_scope + [cand])
            if can_src_to_cand and can_cand_to_dst:
                aff1 = self.calculate_edge_affinity(src_cell, cand, orchestrator)
                aff2 = self.calculate_edge_affinity(cand, dst_cell, orchestrator)
                score = aff1 + aff2
                if resolves_property_mismatch:
                    score += 5.0
                    req_in = sum(1 for p in cand.inputs.values() if p.required)
                    opt_in = len(cand.inputs) - req_in
                    if req_in == 1 and opt_in == 0:
                        score += 3.0  # Clean unary element-wise transform preserving element count
                    elif any("transform_length" in str(getattr(getattr(p, "signature", p), "state", "")) for p in cand.inputs.values()):
                        score -= 2.0  # Penalize transforms requiring output length specifications
                if score > best_score:
                    best_score = score
                    best_b = cand

        if best_b:
            b_inst = best_b.clone() if hasattr(best_b, "clone") else best_b
            b_inst.clause_literals = getattr(dst_cell, "clause_literals", None)
            b_inst.matched_clause_idx = getattr(dst_cell, "matched_clause_idx", None)
            return b_inst

        # Open-World Continuum (Issue 5): If no pre-harvested bridge exists in the lattice,
        # instantiate a TypedHoleCell(tau_src -> tau_dst) when synthesis is available.
        if best_b is None and src_cell.primary_output and dst_cell.primary_input:
            try:
                from dag_plan import TypedHoleCell
                from inference import ModelManager
                if ModelManager.get_instance().can_synthesize():
                    hole_cell = TypedHoleCell(
                        hole_id=f"HOLE_BRIDGE_{src_cell.cell_id}_TO_{dst_cell.cell_id}",
                        task_description=f"Bridge {src_cell.primary_output.type_name} to {dst_cell.primary_input.type_name}",
                        in_sig=src_cell.primary_output.signature,
                        out_sig=dst_cell.primary_input.signature,
                    )
                    hole_cell.matched_clause_idx = getattr(dst_cell, "matched_clause_idx", None)
                    hole_cell.clause_literals = getattr(dst_cell, "clause_literals", None)
                    return hole_cell
            except Exception as hole_exc:
                logger.debug("[ROUTE] TypedHoleCell bridge fallback skipped: %s", hole_exc)
        return None

    @classmethod
    def extract_file_literals(cls, prompt: str, ctx: Optional[Any] = None) -> Tuple[List[str], List[str]]:
        """
        Extracts source and destination file literals from prompt and/or ExecutionContext.
        Preserves typed intent across layers without brittle re-scanning.
        """
        if ctx:
            ctx_src = list(getattr(ctx, "parameters", {}).get("source_uris") or getattr(ctx, "source_files", []) or [])
            ctx_dest = list(getattr(ctx, "parameters", {}).get("dest_uris") or getattr(ctx, "dest_files", []) or [])
            if ctx_src or ctx_dest:
                return ctx_src, ctx_dest

        extracted = ExecutionContext._extract_universal_literals(prompt or "")
        src_lits = [
            val for pos, kind, val in extracted
            if (kind == "file_asset" or (kind == "quoted_str" and ExecutionContext._is_path_string(str(val))))
            and ExecutionContext._asset_direction(prompt or "", pos) != "dest"
        ]
        dest_lits = [
            val for pos, kind, val in extracted
            if (kind == "file_asset" or (kind == "quoted_str" and ExecutionContext._is_path_string(str(val))))
            and ExecutionContext._asset_direction(prompt or "", pos) == "dest"
        ]
        return src_lits, dest_lits

    @classmethod
    def segment_prompt_clauses(cls, prompt: str) -> List[str]:
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

    @classmethod
    def tag_cells_with_clause_indices(cls, path: List[Cell], prompt: str, orchestrator: Any = None) -> List[Dict[str, Any]]:
        """
        Tag path cells with the prompt clause they SERVE (matched_clause_idx).

        Evidence comes from token_evidence (corpus IDF, identity precision x
        recall). A clause is served by the path cell(s) with the best identity
        F-score for it; a cell that is merely a weaker partial match for a clause
        another path cell explains better stays clause-neutral, so downstream
        dead-output and coverage checks can see it is unrequested. Returns the
        decision table (also used by the --debug coverage panel).
        """
        decisions: List[Dict[str, Any]] = []
        if not path or not prompt:
            return decisions
        from token_evidence import get_evidence
        orch = orchestrator
        if orch is None:
            for c in path:
                orch = getattr(c, "_orchestrator", None) or orch
        if orch is None:
            orch = _ACTIVE_ORCHESTRATOR[0] if _ACTIVE_ORCHESTRATOR else None
        ev = get_evidence(orch)
        clauses = cls.segment_prompt_clauses(prompt)
        if ev is None or not clauses:
            return decisions
        clause_toks = [ev.clause_tokens(cl) for cl in clauses]
        # F matrix: path cell x clause
        table = []
        for cell in path:
            row = [ev.identity_overlap(ct, cell) if ct else {"f": 0.0, "hit": 0.0, "rank": (0.0, 0.0, 0.0), "tokens": []} for ct in clause_toks]
            table.append(row)
        winner_f = []
        for j in range(len(clauses)):
            winner_f.append(max((table[i][j]["rank"] for i in range(len(path))), default=(0.0, 0.0, 0.0)))
        for i, cell in enumerate(path):
            if getattr(cell, "matched_clause_idx", None) is not None:
                continue
            best_j, best_f = None, (0.0, 0.0, 0.0)
            for j in range(len(clauses)):
                f = table[i][j]["rank"]
                if table[i][j]["hit"] > 0.0 and f >= winner_f[j] and f > best_f:
                    best_j, best_f = j, f
            if best_j is None:
                # Not the winner anywhere: still serves a clause if it explains tokens the winner left over
                # (e.g. "train a model ... to predict": predict wins, fit explains the rest). Same rule as
                # the planner's secondary anchors and the coverage audit.
                best_res = (0.0, 0.0, 0.0)
                for j in range(len(clauses)):
                    if table[i][j]["hit"] <= 0.0 or not clause_toks[j]:
                        continue
                    w_tokens = set()
                    for k in range(len(path)):
                        if table[k][j]["rank"] == winner_f[j]:
                            w_tokens |= set(table[k][j]["tokens"])
                    res = ev.identity_overlap(clause_toks[j] - w_tokens, cell)
                    if res["hit"] > 0.0 and res["rank"] > best_res:
                        best_j, best_res = j, res["rank"]
            if best_j is not None:
                cell.matched_clause_idx = best_j
            decisions.append({
                "cell": cell.cell_id, "tagged_clause": best_j,
                "best_rank": tuple(round(x, 3) for x in best_f),
                "outranked_on": [j for j in range(len(clauses))
                                 if table[i][j]["hit"] > 0.0 and table[i][j]["rank"] < winner_f[j]],
            })
        # A connector (zero identity evidence anywhere: a bridge/cast/property-establishing cell) is part of the
        # realisation of the cell it extends, so it inherits that cell's clause. Without a clause it lost every
        # adjacency contest and its output was routed to the wrong consumer.
        for i, cell in enumerate(path):
            if i > 0 and getattr(cell, "matched_clause_idx", None) is None and \
                    getattr(path[i - 1], "matched_clause_idx", None) is not None:
                cell.matched_clause_idx = path[i - 1].matched_clause_idx
        # Clause-scoped literals (column names, quoted strings) travel with the clause a cell
        # serves. Only M1/M6 used to attach them, so literal-scoped binding (target column,
        # feature columns, provenance gates) was silently off for every other method.
        try:
            from unification import ExecutionContext as _EC
            for cell in path:
                j = getattr(cell, "matched_clause_idx", None)
                if j is None or j >= len(clauses) or getattr(cell, "clause_literals", None):
                    continue
                cell.clause_literals = [v for _, k, v in _EC._extract_universal_literals(clauses[j])
                                        if k in ("identifier", "quoted_str")]
        except Exception as _lit_err:
            decisions.append({"cell": None, "error": f"clause literal attachment failed: {_lit_err}"})
        return decisions


    def _find_type_producer(self, anchor, routed_path, candidates, clauses, orch):
        """Type-directed repair: the anchor needs a type no cell in scope produces
        (e.g. predict needs a Regressor). Pick the candidate that (1) has identity
        evidence on the SAME prompt clause, (2) unifies from the current scope and
        (3) unifies into the anchor. Ranked by the shared evidence rank. No
        vocabulary, no thresholds: a producer must be demanded by types AND named
        by the clause."""
        from token_evidence import get_evidence
        ev = get_evidence(orch or self.orchestrator)
        idx = getattr(anchor, "matched_clause_idx", None)
        if idx is None and clauses:
            try:
                idx = max(range(len(clauses)), key=lambda i: self.clause_fit(clauses[i], anchor))
            except Exception:
                idx = len(clauses) - 1
        if ev is None or idx is None or not clauses or not routed_path:
            return None
        if idx >= len(clauses):
            idx = len(clauses) - 1
        ctoks = ev.clause_tokens(clauses[idx])
        in_path = {c.cell_id for c in routed_path}
        best, best_rank = None, None
        for cand in candidates:
            if cand.cell_id in in_path or cand.cell_id == anchor.cell_id:
                continue
            o = ev.identity_overlap(ctoks, cand)
            if o["hit"] <= 0.0:
                continue
            if not (self.step_unifies(routed_path[-1], cand, prev_path=routed_path)
                    or self.step_unifies_dag(cand, routed_path)):
                continue
            if not self.step_unifies(cand, anchor, prev_path=routed_path + [cand]):
                continue
            if best_rank is None or o["rank"] > best_rank:
                best, best_rank = cand, o["rank"]
        if best is None:
            return None
        inst = best.clone() if hasattr(best, "clone") else best
        inst.matched_clause_idx = idx
        inst.clause_literals = getattr(anchor, "clause_literals", {})
        return inst


    def chain_is_valid(self, path: List[Cell]) -> bool:
        """Every step unifies with its predecessor or with the DAG scope of the prefix."""
        for i in range(1, len(path)):
            if not (self.step_unifies(path[i - 1], path[i], prev_path=path[:i])
                    or self.step_unifies_dag(path[i], path[:i])):
                return False
        return True

    def _find_lens_commit_cell(self, cell: Cell, reg: Any) -> Optional[Cell]:
        if self.orchestrator is None:
            return None
        commit_ids = {
            str(e).upper()
            for key in ("component_commit_cells", "column_commit_cell")
            for e in (reg.get_verification_semantics(key) or ())
        }
        loaded = getattr(self.orchestrator, "loaded_cells", {}) or {}
        if commit_ids:
            for c_id, c in loaded.items():
                if c_id.upper() in commit_ids:
                    return c
        target_domain = getattr(cell, "domain_name", "") or ""
        carrier_bind_tags = {"data_carrier", "receiver", "carrier"}
        key_bind_tags = {"key", "column_key", "field_key", "attr_key"}
        candidates = []
        for c in loaded.values():
            if target_domain and getattr(c, "domain_name", "") != target_domain:
                continue
            inputs = getattr(c, "inputs", {}) or {}
            has_carrier = any(getattr(p, "binds", None) in carrier_bind_tags or getattr(p, "port_role", None) in ("receiver", "carrier") for p in inputs.values())
            has_key = any(getattr(p, "binds", None) in key_bind_tags for p in inputs.values())
            has_val = any(getattr(p, "binds", None) == "assigned_value" for p in inputs.values())
            if has_carrier and has_key and has_val:
                candidates.append(c)
        if candidates:
            return candidates[0]
        for c in loaded.values():
            inputs = getattr(c, "inputs", {}) or {}
            has_carrier = any(getattr(p, "binds", None) in carrier_bind_tags or getattr(p, "port_role", None) in ("receiver", "carrier") for p in inputs.values())
            has_key = any(getattr(p, "binds", None) in key_bind_tags for p in inputs.values())
            has_val = any(getattr(p, "binds", None) == "assigned_value" for p in inputs.values())
            if has_carrier and has_key and has_val:
                return c
        return None

    def _get_setter_carrier_type(self, setter: Cell) -> Optional[str]:
        carrier_bind_tags = {"data_carrier", "receiver", "carrier"}
        for p in (getattr(setter, "inputs", {}) or {}).values():
            if getattr(p, "binds", None) in carrier_bind_tags or getattr(p, "port_role", None) in ("receiver", "carrier"):
                return str(getattr(getattr(p, "signature", p), "type_name", "") or "").lower()
        return None

    def _consumes_carrier(self, cell: Cell, carrier_type: Optional[str], reg: Any) -> bool:
        if not carrier_type:
            return False
        for p in (getattr(cell, "inputs", {}) or {}).values():
            tn = str(getattr(getattr(p, "signature", p), "type_name", "") or "").lower()
            if tn and (tn == carrier_type or reg.is_subtype(tn, carrier_type) or reg.is_subtype(carrier_type, tn)):
                return True
            ab = str(getattr(getattr(p, "signature", p), "abstract_type", "") or "").lower()
            if ab and reg.is_subtype(carrier_type, ab):
                return True
        return False

    def _consumes_table(self, cell: Cell) -> bool:
        try:
            from lattice import TypeRegistry
            reg = TypeRegistry.get_instance()
            return self._consumes_carrier(cell, "table", reg)
        except Exception:
            return False

    def insert_column_commits(self, path: List[Cell], trace: bool = True) -> List[Cell]:
        """
        Universal Lens Commit Lifting:
        Commits projected component transformations back into their parent composite carrier.

        Any morphism that transforms a projected component of a composite carrier
        (cell.effects intersecting verification_semantics component_transform_effects / column_transform_effects)
        produces a new component value. When a downstream morphism consumes the parent
        composite carrier C, inserts the domain's declared setter morphism μ: C x K x V -> C
        right after it, scoped to the same clause and literal.
        """
        try:
            from lattice import TypeRegistry
            reg = TypeRegistry.get_instance()
            effects = {
                str(e).lower()
                for key in ("component_transform_effects", "column_transform_effects")
                for e in (reg.get_verification_semantics(key) or ())
            }
        except Exception:
            return path
        if not effects or self.orchestrator is None:
            return path

        out = list(path)
        i = 0
        while i < len(out):
            c = out[i]
            lits = getattr(c, "clause_literals", None)
            c_fx = {str(e).lower() for e in (getattr(c, "effect_names", lambda: [])() or getattr(c, "effects", None) or [])}
            if (c_fx & effects) and lits:
                proto = self._find_lens_commit_cell(c, reg)
                if proto is not None:
                    nxt_is_commit = i + 1 < len(out) and out[i + 1].cell_id == proto.cell_id
                    carrier_tn = self._get_setter_carrier_type(proto) or "table"
                    if not nxt_is_commit and any(self._consumes_carrier(x, carrier_tn, reg) for x in out[i + 1:]):
                        commit = proto.clone() if hasattr(proto, "clone") else proto
                        commit.matched_clause_idx = getattr(c, "matched_clause_idx", None)
                        commit.clause_literals = list(lits)
                        trial = out[: i + 1] + [commit] + out[i + 1:]
                        if self.chain_is_valid(trial):
                            if trace:
                                self._trace("lens_commit_inserted", after=c.cell_id, commit=commit.cell_id, key=list(lits),
                                            carrier=carrier_tn, reason="component transform is followed by carrier-consuming cell")
                            out = trial
                            i += 1
                        else:
                            if trace:
                                self._trace("lens_commit_blocked", after=c.cell_id, commit=proto.cell_id,
                                            reason="chain would not unify with the lens commit cell inserted")
            i += 1
        try:
            return OpenPortFrontier.build_dag_plan(out)
        except Exception:
            return out

    def insert_property_bridges(
        self,
        path: List[Cell],
        prompt: str = "",
        ctx: Optional[Any] = None,
        trace: bool = True,
    ) -> List[Cell]:
        """
        Inserts property-resolving bridges (e.g. converting complex frequency outputs to real
        magnitudes via NUMPY_ABS) when an upstream cell produces a property that conflicts
        with a downstream consumer's declared preconditions or column contract.
        """
        if not path or self.orchestrator is None:
            return path
        try:
            from property_contracts import analyze as _prop_analyze
            from unification import UnificationGate, ExecutionContext
            resolved_prompt = prompt or getattr(self, "_current_prompt", "") or getattr(self, "prompt", "")
            if ctx is not None:
                resolved_ctx = ctx
                if resolved_prompt and hasattr(resolved_ctx, "prompt") and not getattr(resolved_ctx, "prompt", ""):
                    resolved_ctx.prompt = resolved_prompt
                elif not resolved_prompt and getattr(resolved_ctx, "prompt", ""):
                    resolved_prompt = resolved_ctx.prompt
            else:
                resolved_ctx = ExecutionContext(prompt=resolved_prompt)

            gate = UnificationGate(orchestrator=self.orchestrator)
            res = gate.unify_pipeline(path, resolved_ctx)
            if res.is_bottom() or not gate.last_pipeline_bindings:
                return path
            issues = _prop_analyze(gate.last_pipeline_bindings, self.orchestrator)
            if not issues:
                return path

            out = list(path)
            for iss in issues:
                prod_id = iss.get("established_by")
                candidates = iss.get("bridge_candidates", [])
                prop = iss.get("property")
                if not prod_id or not candidates:
                    continue
                prod_idx = next((i for i, c in enumerate(out) if c.cell_id == prod_id), None)
                if prod_idx is None:
                    continue

                # Advance past any intermediate cells that preserve the offending property
                while prod_idx + 1 < len(out) and any(
                    str(eff).lower() == f"preserves:{str(prop).lower()}"
                    for eff in (getattr(out[prod_idx + 1], "effects", None) or [])
                ):
                    prod_idx += 1

                best_cand = None
                for cid in candidates:
                    cand_cell = self.orchestrator.loaded_cells.get(cid)
                    if cand_cell:
                        if len(cand_cell.inputs) == 1:
                            best_cand = cand_cell
                            break
                        if best_cand is None:
                            best_cand = cand_cell

                if best_cand:
                    if prod_idx + 1 < len(out) and out[prod_idx + 1].cell_id == best_cand.cell_id:
                        continue
                    b_cell = best_cand.clone() if hasattr(best_cand, "clone") else best_cand
                    b_cell.matched_clause_idx = getattr(out[prod_idx], "matched_clause_idx", None)

                    # Wire b_cell to receive output from prod_cell
                    b_out_name = getattr(b_cell, "primary_out", None) or (
                        list(b_cell.outputs.keys())[0] if getattr(b_cell, "outputs", None) else "output_var"
                    )
                    b_in_name = getattr(b_cell, "primary_in", None) or (
                        list(b_cell.inputs.keys())[0] if getattr(b_cell, "inputs", None) else None
                    )
                    prod_cell = out[prod_idx]
                    prod_out_name = getattr(prod_cell, "primary_out", None) or (
                        list(prod_cell.outputs.keys())[0] if getattr(prod_cell, "outputs", None) else "output_var"
                    )
                    if b_in_name:
                        b_cell.planned_port_wires = {b_in_name: (prod_cell, prod_out_name)}
                        b_cell.bound_parent_ids = {prod_id}

                    trial = out[: prod_idx + 1] + [b_cell] + out[prod_idx + 1 :]

                    # Rewire downstream consumers in trial[prod_idx + 2:] that were wired to prod_id
                    for downstream in trial[prod_idx + 2 :]:
                        dw_wires = getattr(downstream, "planned_port_wires", None)
                        if isinstance(dw_wires, dict):
                            new_wires = {}
                            for p_name, tgt in dw_wires.items():
                                if isinstance(tgt, tuple) and len(tgt) == 2:
                                    t_cell, t_port = tgt
                                    t_cid = getattr(t_cell, "cell_id", str(t_cell))
                                    if t_cid == prod_id:
                                        new_wires[p_name] = (b_cell, b_out_name)
                                    else:
                                        new_wires[p_name] = tgt
                                else:
                                    t_cid = getattr(tgt, "cell_id", str(tgt))
                                    if t_cid == prod_id:
                                        new_wires[p_name] = (b_cell, b_out_name)
                                    else:
                                        new_wires[p_name] = tgt
                            downstream.planned_port_wires = new_wires
                        dw_parents = getattr(downstream, "bound_parent_ids", None)
                        if isinstance(dw_parents, set) and prod_id in dw_parents:
                            dw_parents.discard(prod_id)
                            dw_parents.add(b_cell.cell_id)

                    if self.chain_is_valid(trial):
                        if trace:
                            self._trace("property_bridge_inserted", after=prod_id, bridge=b_cell.cell_id,
                                        property=iss.get("property"), reason="resolves downstream property precondition")
                        out = trial
            try:
                return OpenPortFrontier.build_dag_plan(out)
            except Exception:
                return out
        except Exception as _p_err:
            logger.debug(f"[ROUTE] insert_property_bridges skipped: {_p_err}")
            return path


    def prune_unrequested(self, path: List[Cell], prompt: str) -> List[Cell]:
        """
        Post-planning pass shared by every method. Remove cells the coverage audit calls UNREQUESTED
        (outranked on every clause they touch, explaining no leftover clause token) when the chain
        stays type-valid without them. Each removal is traced. Never removes the first cell.
        """
        from coverage_audit import audit
        current = list(path)
        for _ in range(len(current)):
            res = audit(current, prompt, self.orchestrator)
            if not res.get("available"):
                break
            victim = next((i for i in res["issues"] if i["kind"] == "unrequested_cell"), None)
            if victim is None:
                break
            idx = next((k for k, c in enumerate(current) if c.cell_id == victim["cell"]), None)
            if idx is None or idx == 0:
                break
            trial = current[:idx] + current[idx + 1:]
            if not self.chain_is_valid(trial):
                self._trace("prune_blocked", cell=victim["cell"], reason="chain would not unify without it",
                            winners=victim["winners"])
                break
            self._trace("pruned_unrequested", cell=victim["cell"], winners=victim["winners"],
                        reason="outranked on every clause it touches and explains no leftover clause token")
            current = trial
        try:
            from dag_plan import OpenPortFrontier
            active_ids = {c.cell_id for c in current}
            for c in current:
                pw = getattr(c, "planned_port_wires", None)
                if isinstance(pw, dict):
                    c.planned_port_wires = {
                        k: v for k, v in pw.items()
                        if getattr(v[0] if isinstance(v, tuple) else v, "cell_id", str(v[0] if isinstance(v, tuple) else v)) in active_ids
                    }
                bp = getattr(c, "bound_parent_ids", None)
                if isinstance(bp, set):
                    c.bound_parent_ids = bp & active_ids
            return OpenPortFrontier.build_dag_plan(current)
        except Exception:
            return current

    # ------------------------------------------------------------------
    # Shared evidence primitives (identical for every route method)
    # ------------------------------------------------------------------
    def _evidence(self) -> Any:
        from token_evidence import get_evidence
        return get_evidence(self.orchestrator)

    def clause_fit(self, clause_text: str, cell: Cell) -> Tuple[float, float, float]:
        """Identity evidence rank (recall, name-precision, F) of `cell` for a prompt clause; zeros if none."""
        ev = self._evidence()
        if ev is None or not clause_text:
            return (0.0, 0.0, 0.0)
        toks = ev.clause_tokens(clause_text)
        if not toks:
            return (0.0, 0.0, 0.0)
        o = ev.identity_overlap(toks, cell)
        return o["rank"] if o["hit"] > 0.0 else (0.0, 0.0, 0.0)

    @staticmethod
    def _is_path_consumer(cell: Cell) -> bool:
        """True if the cell reads/writes a filesystem path (tree-declared roles / abstract types)."""
        try:
            from lattice import TypeRegistry
            reg = TypeRegistry.get_instance()
            roles = {str(r).lower() for r in reg.get_verification_semantics("path_consumer_roles")}
            abstracts = {str(r).lower() for r in reg.get_verification_semantics("path_abstract_types")}
        except Exception:
            roles, abstracts = set(), set()
        for p in cell.inputs.values():
            sig = getattr(p, "signature", p)
            if str(getattr(p, "abstract_type", "") or "").lower() in abstracts or str(getattr(sig, "abstract_type", "") or "").lower() in abstracts:
                return True
            if str(getattr(p, "port_role", "") or "").lower() in roles or str(getattr(p, "derived_role", "") or "").lower() in roles:
                return True
        return False

    def file_compat(self, cell: Cell, literal: Any) -> int:
        """+1 format-compatible, -1 incompatible, 0 not applicable (no literal / not a path consumer)."""
        if literal is None or not self._is_path_consumer(cell):
            return 0
        return 1 if self.is_file_format_compatible(cell, str(literal)) else -1

    def requested_sink(self, prompt: str, candidates: List[Cell]) -> Optional[Cell]:
        """
        A sink (egress/terminal) cell is requested only if some clause is better explained by a sink
        than by ANY non-sink candidate. Replaces the "prompt contains the word write" egress guess,
        which turned "write it to a new column" into a file sink.
        """
        ev = self._evidence()
        if ev is None:
            return None
        from unification import _is_sink_cell
        sinks = [c for c in candidates if _is_sink_cell(c)]
        others = [c for c in candidates if not _is_sink_cell(c)]
        best, best_rank = None, (0.0, 0.0, 0.0)
        for cl in self.segment_prompt_clauses(prompt):
            toks = ev.clause_tokens(cl)
            if not toks:
                continue
            top_other = max((ev.identity_overlap(toks, c)["rank"] for c in others), default=(0.0, 0.0, 0.0))
            for s in sinks:
                o = ev.identity_overlap(toks, s)
                if o["hit"] > 0.0 and o["rank"] > top_other and o["rank"] > best_rank:
                    best, best_rank = s, o["rank"]
        return best

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
            if getattr(c, "node_type", "") == "constant":
                continue
            if not getattr(c, "slots", None):
                if cid.startswith("MACRO_"):
                    continue
                if getattr(c, "is_combinator", False) or getattr(c, "node_role", "") == "combinator" or getattr(c, "role", "") == "combinator" or getattr(c, "node_type", "") == "combinator":
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

                # Format/modality verification from clause literal
                cl_file_ext = None
                for w in cl.split():
                    clean_w = w.strip(" '\"`(),;:")
                    if "." in clean_w and not clean_w.startswith("."):
                        ext = clean_w.rsplit(".", 1)[1].lower()
                        if ext and len(ext) <= 5 and ext.isalnum():
                            cl_file_ext = ext
                            break

                if cl_file_ext:
                    if not self.is_file_format_compatible(c, f"file.{cl_file_ext}"):
                        score = 0.0

                # Estimator mathematical contract verification via declared role semantics
                try:
                    from signature_introspector import introspect_estimator_role
                    from tokenizer import normalize_token
                    role = introspect_estimator_role(c)
                    if role:
                        from lattice import TypeRegistry
                        reg = TypeRegistry.get_instance()
                        declared_roles = {
                            str(r).lower() for r in (reg.get_verification_semantics("model_roles") or [])
                        } | {"regressor", "classifier", "clusterer"}
                        target_families = {"regressor", "classifier", "clusterer"} & declared_roles
                        norm_cl_toks = {normalize_token(t) for t in cl_toks}
                        matched_families = {
                            fam for fam in target_families
                            if any(t.startswith("classif" if fam == "classifier" else normalize_token(fam)) for t in norm_cl_toks)
                        }
                        if len(matched_families) == 1:
                            target_family = next(iter(matched_families))
                            if role in target_families and role != target_family:
                                score = 0.0
                except Exception:
                    pass

                if score <= 0.0:
                    continue
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
            if getattr(c, "is_combinator", False) or getattr(c, "node_role", "") == "combinator" or getattr(c, "role", "") == "combinator" or getattr(c, "node_type", "") == "combinator":
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

