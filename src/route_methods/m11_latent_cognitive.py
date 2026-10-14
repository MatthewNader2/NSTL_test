"""
src/route_methods/m11_latent_cognitive.py - Neuro-Symbolic Topological Lattice (NSTL)

M11: Cognitive Latent Pathfinding over Lattice DAG.
Combines category-theoretic type-unification constraints with continuous
latent residual stream projections extracted from local sub-1B models
(EmbeddingGemma-300M / Qwen2.5-Coder-0.5B).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np

from .base import RouteMethod

try:
    from ..lattice import Cell, LatticeOrchestrator, TypeRegistry
    from ..tokenizer import CellTokenizer
    from ..unification import ExecutionContext
    from ..planner import _active_vocab
    from ..cognitive_pathfinder import (
        CognitivePathfinder,
        ModelUnavailableError,
        get_latent_scoring_weights,
    )
    from ..log_config import get_logger
except (ImportError, ValueError):
    from lattice import Cell, LatticeOrchestrator, TypeRegistry
    from tokenizer import CellTokenizer
    from unification import ExecutionContext
    from planner import _active_vocab
    from cognitive_pathfinder import (
        CognitivePathfinder,
        ModelUnavailableError,
        get_latent_scoring_weights,
    )
    from log_config import get_logger

logger = get_logger("m11_latent_cognitive")


def _get_sink_roles(orchestrator: Optional[LatticeOrchestrator] = None) -> Set[str]:
    """
    Dynamically queries sink/terminal roles from TypeRegistry and active lattice
    vocabulary—zero hardcoded role literals.
    """
    roles: Set[str] = set()
    try:
        reg = TypeRegistry.get_instance()
        for key in ("sink_roles", "terminal_roles", "egress_roles"):
            for r in reg.get_verification_semantics(key):
                if r:
                    roles.add(str(r).strip().lower())
    except Exception as e:
        logger.debug("[M11] TypeRegistry sink_roles lookup error: %s", e)

    try:
        vocab = _active_vocab(orchestrator)
        roles.update(str(r).lower() for r in vocab.terminal_roles if r)
    except Exception as e:
        logger.debug("[M11] LatticeVocabulary terminal_roles lookup error: %s", e)

    return roles


def _is_sink_node(cell: Any, orchestrator: Optional[LatticeOrchestrator] = None) -> bool:
    """
    Determines whether a cell is a terminal sink using declared cell properties,
    TypeRegistry verification semantics, and derived lattice stage structure.
    """
    if getattr(cell, "is_sink", False):
        return True
    if getattr(cell, "endable", None) is True:
        return True

    role = str(getattr(cell, "node_role", "") or getattr(cell, "role", "")).strip().lower()
    sink_roles = _get_sink_roles(orchestrator)
    if role and role in sink_roles:
        return True

    try:
        vocab = _active_vocab(orchestrator)
        if vocab.terminal_stage is not None and getattr(cell, "stage", None) == vocab.terminal_stage:
            return True
    except Exception as e:
        logger.debug("[M11] Terminal stage check fallback: %s", e)

    return bool(getattr(cell, "is_endable", False))


def _is_ingress_node(cell: Any, orchestrator: Optional[LatticeOrchestrator] = None) -> bool:
    """Determines whether a cell belongs to the lattice's declared ingress stage."""
    if getattr(cell, "is_ingress", False):
        return True
    try:
        vocab = _active_vocab(orchestrator)
        if vocab.ingress_stage is not None and getattr(cell, "stage", None) == vocab.ingress_stage:
            return True
        role = str(getattr(cell, "node_role", "") or getattr(cell, "role", "")).strip().lower()
        if role and role in vocab.ingress_roles:
            return True
    except Exception as e:
        logger.debug("[M11] Ingress stage check fallback: %s", e)
    return False


class M11LatentCognitiveRouteMethod(RouteMethod):
    """
    RouteMethod M11: Cognitive Latent Pathfinding.
    Evaluates candidate morphisms by projecting directional progress vectors
    in the latent representation space of local sub-1B models toward the prompt goal,
    strictly constrained by algebraic typestate unification gates.
    """
    name: str = "m11_latent_cognitive"

    def __init__(self, orchestrator: Optional[LatticeOrchestrator] = None, **kwargs):
        super().__init__(orchestrator=orchestrator, **kwargs)
        self.pathfinder = CognitivePathfinder(extractor=kwargs.get("extractor"))

    def is_available(self) -> bool:
        """Returns True iff the underlying neural latent extractor is loaded and ready."""
        return self.pathfinder.is_available()

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
        self._current_prompt = prompt
        orch = orchestrator or self.orchestrator
        if not self.is_available():
            err_msg = (
                "M11 Cognitive Latent Pathfinding requires a loaded neural embedding or GGUF model; "
                "refusing to execute with pseudo-random noise."
            )
            self.fallback_reason = err_msg
            raise ModelUnavailableError(err_msg)

        if not tunnel:
            return []
        if len(tunnel) == 1:
            return [tunnel[0]]

        candidates = [
            c for c in tunnel
            if getattr(c, "node_type", "") != "constant"
            and (
                bool(getattr(c, "slots", None))
                or (
                    getattr(c, "node_type", "") != "macro"
                    and not c.cell_id.startswith("MACRO_")
                    and not (
                        getattr(c, "is_combinator", False)
                        or getattr(c, "node_role", "") == "combinator"
                        or getattr(c, "role", "") == "combinator"
                        or getattr(c, "node_type", "") == "combinator"
                    )
                )
            )
        ]
        if not candidates:
            return [tunnel[0]]

        # Experimental Lab Priors: ZCA Whitening + SAE Sparse Coding + Absorbing Markov Chain
        self.pathfinder.ensure_corpus_whitening(candidates)
        sae_boosts = self.pathfinder.compute_sparse_concept_boosts(prompt, candidates)
        markov_potentials: Dict[str, float] = {}
        try:
            try:
                from ..experiments import AbsorbingMarkovRouter, ACTIVE_EXPERIMENTS
            except (ImportError, ValueError):
                from experiments import AbsorbingMarkovRouter, ACTIVE_EXPERIMENTS
            if ACTIVE_EXPERIMENTS.use_markov_hitting_prior:
                markov_potentials = AbsorbingMarkovRouter.compute_node_potentials(
                    candidates=candidates,
                    relevance_map=relevance_map,
                    edge_affinity_fn=lambda u, v: self.calculate_edge_affinity(u, v, orch, relevance_map=relevance_map),
                    unify_check_fn=lambda u, v: self.step_unifies(u, v),
                    is_sink_fn=lambda c: _is_sink_node(c, orch),
                    temperature=ACTIVE_EXPERIMENTS.markov_temperature,
                )
        except Exception as mk_exc:
            logger.debug("[M11] Markov potential computation skipped: %s", mk_exc)

        clauses = self.segment_prompt_clauses(prompt)
        if not clauses:
            clauses = [prompt]

        weights = get_latent_scoring_weights()
        w_cl = weights.get("clause_weight", 0.5)
        w_goal = weights.get("goal_weight", 0.3)
        w_struct = weights.get("structural_weight", 0.2)
        w_total = (w_cl + w_goal + w_struct) or 1.0
        momentum = weights.get("state_momentum", 0.7)

        # 1. Continuous Latent Goal and Clause Representations (fails loudly if model unavailable)
        goal_vec = self.pathfinder.extractor.encode(prompt)
        clause_vecs = self.pathfinder.extractor.encode_batch(clauses)
        current_vec = np.zeros_like(goal_vec)

        src_file_literals, dest_file_literals = self.extract_file_literals(prompt or "", ctx=ctx)
        src_lit = src_file_literals[0] if src_file_literals else None
        dst_lit = dest_file_literals[0] if dest_file_literals else None

        # 2. Select initial entry cell C_0 using derived ingress stage
        ingress_cands = [c for c in candidates if _is_ingress_node(c, orch)]
        entry_pool = ingress_cands if ingress_cands else candidates[:5]

        first_target_vec = clause_vecs[0] if clause_vecs else goal_vec
        best_entry = max(
            entry_pool,
            key=lambda c: (
                self.file_compat(c, src_lit),
                self.pathfinder.compute_latent_heuristic(
                    current_vec, self.pathfinder.get_cell_vector(c), first_target_vec, weights=weights
                ),
                relevance_map.get(c.cell_id, 0.0),
            ),
        )

        committed_path: List[Cell] = [best_entry]
        visited_ids: Set[str] = {best_entry.cell_id}
        current_vec = self.pathfinder.get_cell_vector(best_entry)

        _ev = self._evidence()
        prompt_tokens = _ev.clause_tokens(prompt) if _ev is not None else CellTokenizer.tokenize_prompt(prompt)
        covered_tokens: Set[str] = set(getattr(best_entry, "identity_tokens", best_entry.token_set) & prompt_tokens)

        num_clauses = len(clauses)
        max_steps = max(32, num_clauses * 4 + 4, max_transforms + 8)

        # 3. Latent Geodesic Path Progression
        for step in range(max_steps):
            curr = committed_path[-1]

            # Terminal check via registry/vocabulary sink semantics
            if _is_sink_node(curr, orch) and len(committed_path) > 1:
                if step >= max(1, num_clauses - 1) or not (prompt_tokens - covered_tokens):
                    break

            valid_next = [
                c for c in candidates
                if c.cell_id not in visited_ids and (
                    self.step_unifies(curr, c, prev_path=committed_path)
                    or self.step_unifies_dag(c, committed_path, ctx=ctx)
                )
            ]

            if not valid_next:
                break

            cl_idx = min(step + 1, num_clauses - 1)
            target_cl_vec = clause_vecs[cl_idx] if cl_idx < len(clause_vecs) else goal_vec
            target_cl = clauses[cl_idx] if cl_idx < num_clauses else prompt
            _requested_sink = self.requested_sink(prompt, candidates)

            def _score_cand(cand: Cell) -> Tuple[Any, ...]:
                c_vec = self.pathfinder.get_cell_vector(cand)
                latent_goal_h = self.pathfinder.compute_latent_heuristic(
                    current_vec, c_vec, goal_vec, weights=weights
                )
                latent_cl_h = self.pathfinder.compute_latent_heuristic(
                    current_vec, c_vec, target_cl_vec, weights=weights
                )
                aff = self.calculate_edge_affinity(curr, cand, orch, relevance_map=relevance_map)
                rel = relevance_map.get(cand.cell_id, 0.0)

                curr_stage = getattr(curr, "stage", None) or 0
                cand_stage = getattr(cand, "stage", None) or 0
                progress = 2 if cand_stage > curr_stage else (1 if cand_stage == curr_stage else 0)

                is_cand_sink = _is_sink_node(cand, orch)
                is_curr_sink = _is_sink_node(curr, orch)
                unrequested_egress = (
                    is_cand_sink and self._is_path_consumer(cand)
                    and not (dst_lit is not None and self.file_compat(cand, dst_lit) > 0)
                ) or (
                    _requested_sink is None
                    and is_cand_sink
                    and not is_curr_sink
                    and self.clause_fit(target_cl, cand)[0] <= 0.0
                )

                sae_bonus = 0.25 * sae_boosts.get(cand.cell_id, 0.0)
                mk_bonus = 0.20 * markov_potentials.get(cand.cell_id, 0.0)
                composite_latent = (
                    (w_cl * latent_cl_h + w_goal * latent_goal_h + w_struct * (aff + rel)) / w_total
                ) + sae_bonus + mk_bonus
                return (0 if unrequested_egress else 1, composite_latent, progress)

            valid_next.sort(key=_score_cand, reverse=True)
            best_next = valid_next[0]

            committed_path.append(best_next)
            visited_ids.add(best_next.cell_id)
            covered_tokens.update(getattr(best_next, "identity_tokens", best_next.token_set) & prompt_tokens)

            # Update running latent state vector on the unit hypersphere
            next_vec = self.pathfinder.get_cell_vector(best_next)
            current_vec = self.pathfinder.update_state_vector(current_vec, next_vec, momentum=momentum)

            if step >= num_clauses - 1:
                try:
                    try:
                        from ..token_evidence import audit as _audit
                    except (ImportError, ValueError):
                        from token_evidence import audit as _audit
                    _res = _audit(committed_path, prompt, orch)
                    if _res.get("available") and not any(i["kind"] == "clause_unserved" for i in _res.get("issues", [])):
                        self._trace("m11_goal_reached", step=step, path_len=len(committed_path))
                        break
                except Exception as e:
                    logger.debug("[M11] Coverage audit check skipped: %s", e)

            if (prompt_tokens - covered_tokens) == set() and _is_sink_node(best_next, orch):
                break

        self._trace("m11_plan_complete", cells=[c.cell_id for c in committed_path])
        return committed_path


__all__ = [
    "M11LatentCognitiveRouteMethod",
    "_is_sink_node",
    "_get_sink_roles",
]
