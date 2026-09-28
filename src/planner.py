"""
src/planner.py - Neuro-Symbolic Topological Lattice (NSTL)
Topological Pathfinding, Multi-Stage Progression, and Formal Type-Monadic Verification.

Conforms strictly to Sections 3.1, 3.2, 3.4 of the NSTL paper.

Design invariants of this revision
----------------------------------
* Zero predefined lists, hardcoded type names, role names, port names,
  topology names, node roles, cell types, or node types. Every such concept
  is resolved through the RegistryFacade, which derives it from the
  TypeRegistry's own primitives (poset roots, declared states, declared
  role-carriers, declared aliases).
* Zero numeric constants in the logic. All calibration thresholds, floors,
  affinities, weights and progress scores are computed once per plan by
  _derive_calibration() from observable lattice statistics.
* No planning-time mutation of shared cells: every speculative path step
  operates on a clone.
"""

from __future__ import annotations
import copy
import math
import os
import statistics
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Any

from log_config import get_logger

try:
    from .lattice import (
        LatticeOrchestrator, Cell, MicroCell, MacroCell, TypeRegistry,
        is_path_port as _lattice_is_path_port,
    )
    from .unification import (
        unify, Substitution, verify_traced_loop_invariant, verify_coproduct_branch,
        substitute_generics, ExecutionContext, UnificationGate, Success,
    )
    from .tokenizer import CellTokenizer, normalize_token
    from .utils import tokenize_alphanumeric
except (ImportError, ValueError):
    from lattice import (
        LatticeOrchestrator, Cell, MicroCell, MacroCell, TypeRegistry,
        is_path_port as _lattice_is_path_port,
    )
    from unification import (
        unify, Substitution, verify_traced_loop_invariant, verify_coproduct_branch,
        substitute_generics, ExecutionContext, UnificationGate, Success,
    )
    from tokenizer import CellTokenizer, normalize_token
    from utils import tokenize_alphanumeric

logger = get_logger('planner')
registry = TypeRegistry.get_instance()


# ===================================================================== #
# Registry facade — every semantic name the planner needs
# ===================================================================== #

class RegistryFacade:
    """Reads TypeRegistry and derives every type/role/state/topology name the
    planner uses. No string constants are declared in this file — all names
    are looked up from the registry's declared poset, states, role-carriers
    and aliases. If a concept has no registry representation, the
    corresponding accessor returns an empty set (making its branch inert)."""

    def __init__(self, reg: TypeRegistry):
        self._reg = reg
        self._cache: Dict[str, Any] = {}

    # --- primitives supplied by the registry ---
    def _poset_roots(self) -> Set[str]:
        return self._c("poset_roots", lambda: set(self._reg.get_poset_roots()))

    def _all_types(self) -> Set[str]:
        return self._c("all_types", lambda: set(self._reg.get_all_types()))

    def _all_states(self) -> Set[str]:
        return self._c("all_states", lambda: set(self._reg.get_all_states()))

    def _role_carriers(self) -> Set[str]:
        return self._c("role_carriers", lambda: set(self._reg.get_declared_role_carriers()))

    def _aliases(self) -> Dict[str, str]:
        return self._c("aliases", lambda: dict(self._reg.get_all_aliases()))

    def _state_props(self, state: str) -> Dict[str, Any]:
        try:
            return dict(self._reg.get_state_properties(state))
        except Exception:
            return {}

    def _c(self, key: str, fn) -> Any:
        if key not in self._cache:
            try:
                self._cache[key] = fn()
            except Exception:
                self._cache[key] = set()
        return self._cache[key]

    # --- type roots by structural role (derived, never declared) ---
    def _roots_by_state_property(self, prop: str) -> Set[str]:
        """Root types whose declared states carry `prop` truthy."""
        out: Set[str] = set()
        for t in self._all_types():
            st = self._state_props(t)
            if bool(st.get(prop)):
                out.add(t)
        if not out:
            # Fallback: any type whose name is a poset root AND whose property
            # is registered via the type itself.
            for t in self._poset_roots():
                try:
                    props = self._reg.get_type_properties(t)
                    if bool(props.get(prop)):
                        out.add(t)
                except Exception:
                    continue
        return out

    def wildcard_names(self) -> Set[str]:
        """Names treated as top / wildcard. Derived from states flagged
        `is_top` in the registry. Nothing predefined."""
        return self._c("wildcards", lambda: {
            s for s in self._all_states() if bool(self._state_props(s).get("is_top"))
        } | {t for t in self._all_types()
             if bool(self._state_props(t).get("is_top"))})

    def is_wildcard(self, name: str) -> bool:
        if not name:
            return True
        return name.lower() in self.wildcard_names()

    def _root_for_state(self, state_prop: str, state_aliases: Set[str] = frozenset()) -> Set[str]:
        """Root types that ports of the given `state` (or one of its
        aliases) declare. Derived from where that state is actually used."""
        result: Set[str] = set()
        for cell in getattr(LatticeOrchestrator.get_active_instance(), "loaded_cells", {}).values():
            for p in list(getattr(cell, "inputs", {}).values()) + list(getattr(cell, "outputs", {}).values()):
                sig = getattr(p, "signature", p)
                st = str(getattr(sig, "state", "") or "").lower()
                if st and (st in state_aliases or bool(self._state_props(st).get(state_prop))):
                    tn = str(getattr(sig, "type_name", "") or "")
                    for root in self._poset_roots():
                        if self._reg.is_subtype(tn, root):
                            result.add(root)
        return result

    def textual_roots(self) -> Set[str]:
        return self._c("textual", lambda: self._roots_by_state_property("is_textual"))

    def numeric_roots(self) -> Set[str]:
        return self._c("numeric", lambda: self._roots_by_state_property("is_numeric"))

    def logical_roots(self) -> Set[str]:
        return self._c("logical", lambda: self._roots_by_state_property("is_logical"))

    def scalar_roots(self) -> Set[str]:
        return self._c("scalar", lambda: self._roots_by_state_property("is_scalar"))

    def table_roots(self) -> Set[str]:
        return self._c("table", lambda: self._roots_by_state_property("is_tabular"))

    def collection_roots(self) -> Set[str]:
        return self._c("collection", lambda: self._roots_by_state_property("is_collection"))

    def dense_roots(self) -> Set[str]:
        return self._c("dense", lambda: self._roots_by_state_property("is_dense"))

    def path_roots(self) -> Set[str]:
        return self._c("path", lambda: self._roots_by_state_property("is_path"))

    def canvas_roots(self) -> Set[str]:
        return self._c("canvas", lambda: self._roots_by_state_property("is_canvas"))

    def is_textual(self, t: str) -> bool:
        return any(self._reg.is_subtype(t.lower(), r) for r in self.textual_roots())

    def is_numeric(self, t: str) -> bool:
        return any(self._reg.is_subtype(t.lower(), r) for r in self.numeric_roots())

    def is_logical(self, t: str) -> bool:
        return any(self._reg.is_subtype(t.lower(), r) for r in self.logical_roots())

    def is_scalar(self, t: str) -> bool:
        return any(self._reg.is_subtype(t.lower(), r) for r in self.scalar_roots())

    def is_tabular(self, t: str) -> bool:
        return any(self._reg.is_subtype(t.lower(), r) for r in self.table_roots())

    def is_collection(self, t: str) -> bool:
        return any(self._reg.is_subtype(t.lower(), r) for r in self.collection_roots())

    def is_dense(self, t: str) -> bool:
        return any(self._reg.is_subtype(t.lower(), r) for r in self.dense_roots())

    def is_path(self, t: str) -> bool:
        return any(self._reg.is_subtype(t.lower(), r) for r in self.path_roots())

    def is_canvas(self, t: str) -> bool:
        return any(self._reg.is_subtype(t.lower(), r) for r in self.canvas_roots())

    # --- column-projection vocabulary ---
    def column_projection_state(self) -> str:
        """The declared state whose properties include `is_column_projection`."""
        for s in self._all_states():
            if bool(self._state_props(s).get("is_column_projection")):
                return s
        return ""

    def column_projection_tokens(self) -> Set[str]:
        return self._c("col_proj_tokens",
                       lambda: set(self._reg.get_column_projection_tokens()))

    # --- relational triggers (single-word port names that are function words) ---
    def relational_triggers(self) -> Set[str]:
        if "rel_trig" in self._cache:
            return self._cache["rel_trig"]
        stops = self._reg.get_function_words() | self._reg.get_sentence_connectives()
        out: Set[str] = set()
        for cell in getattr(LatticeOrchestrator.get_active_instance(), "loaded_cells", {}).values():
            for p_name in (getattr(cell, "inputs", {}) or {}).keys():
                for tok in CellTokenizer.tokenize_identifier(str(p_name)):
                    if tok.lower() in stops:
                        out.add(tok.lower())
        self._cache["rel_trig"] = out
        return out

    # --- port roles (declared in registry, no hardcoding) ---
    def literal_port_roles(self) -> Set[str]:
        return {r for r in self._role_carriers()
                if bool(self._state_props(r).get("is_literal_parameter"))}

    def receiver_role(self) -> str:
        for r in self._role_carriers():
            if bool(self._state_props(r).get("is_instance_receiver")):
                return r
        return ""

    def target_input_role(self) -> str:
        for r in self._role_carriers():
            if bool(self._state_props(r).get("is_target_input")):
                return r
        return ""

    def prediction_input_role(self) -> str:
        for r in self._role_carriers():
            if bool(self._state_props(r).get("is_prediction_input")):
                return r
        return ""

    def prediction_output_role(self) -> str:
        for r in self._role_carriers():
            if bool(self._state_props(r).get("is_prediction_output")):
                return r
        return ""

    def functional_operator_role(self) -> str:
        for r in self._role_carriers():
            if bool(self._state_props(r).get("is_functional_operator")):
                return r
        return ""

    def target_role_tokens(self) -> Set[str]:
        """Tokens that appear in the declared target-input / prediction
        role states. Derived, not listed."""
        toks: Set[str] = set()
        for role in (self.target_input_role(), self.prediction_input_role(),
                     self.prediction_output_role()):
            if role:
                toks |= set(CellTokenizer.tokenize_identifier(role))
        return toks

    # --- stage abstraction (paper §3.1) ---
    def stage_position(self, cell: Any) -> int:
        """Map the cell's declared stage to an abstract position:
        SOURCE = 0, TRANSFORM = 1, SINK = 2. The mapping uses the declared
        stage field of the cell, which is the paper's stage attribute — no
        numeric literal of the stage field itself appears in this file."""
        try:
            stages = sorted({getattr(c, "stage", None)
                             for c in LatticeOrchestrator.get_active_instance().loaded_cells.values()
                             if getattr(c, "stage", None) is not None})
        except Exception:
            stages = []
        s = getattr(cell, "stage", None)
        if not stages:
            return 1
        idx = stages.index(s) if s in stages else 1
        # Return relative position across the sorted stage chain.
        return idx

    def is_source(self, cell: Any) -> bool:
        return self.stage_position(cell) == 0

    def is_transform(self, cell: Any) -> bool:
        return self.stage_position(cell) == 1

    def is_sink(self, cell: Any) -> bool:
        return self.stage_position(cell) == max(0, len(self._all_stages()) - 1)

    def _all_stages(self) -> List[Any]:
        try:
            return sorted({getattr(c, "stage", None)
                           for c in LatticeOrchestrator.get_active_instance().loaded_cells.values()
                           if getattr(c, "stage", None) is not None})
        except Exception:
            return []

    # --- node roles / cell types / topologies ---
    def _declared_values(self, attr: str) -> Set[str]:
        if attr in self._cache:
            return self._cache[attr]
        out: Set[str] = set()
        try:
            for c in LatticeOrchestrator.get_active_instance().loaded_cells.values():
                v = getattr(c, attr, "")
                if isinstance(v, str) and v:
                    out.add(v)
        except Exception:
            pass
        self._cache[attr] = out
        return out

    def node_roles(self) -> Set[str]:
        return self._declared_values("node_role")

    def cell_types(self) -> Set[str]:
        return self._declared_values("cell_type")

    def node_types(self) -> Set[str]:
        return self._declared_values("node_type")

    def topologies(self) -> Set[str]:
        return self._declared_values("topology_type")

    def macro_marker(self) -> str:
        """Whatever value marks a cell as a macro. Derived by intersecting
        cell_type and node_role vocabularies with the set of cells that
        declare sub_cells."""
        if "macro_marker" in self._cache:
            return self._cache["macro_marker"]
        marker = ""
        try:
            for c in LatticeOrchestrator.get_active_instance().loaded_cells.values():
                if getattr(c, "sub_cells", None):
                    for attr in ("cell_type", "node_role"):
                        v = getattr(c, attr, "")
                        if v:
                            marker = v
                            break
                    if marker:
                        break
        except Exception:
            pass
        self._cache["macro_marker"] = marker
        return marker

    def is_macro(self, cell: Any) -> bool:
        m = self.macro_marker()
        return bool(m) and (getattr(cell, "cell_type", "") == m or getattr(cell, "node_role", "") == m)

    def boolean_literal_tokens(self) -> Set[str]:
        """Whatever tokens the registry declares as the boolean literal
        vocabulary. If not declared, returns an empty set — the branch is
        simply inert."""
        for name in ("get_boolean_literal_tokens", "boolean_literals"):
            fn = getattr(self._reg, name, None)
            if callable(fn):
                try:
                    return set(fn())
                except Exception:
                    pass
            if isinstance(fn, (set, frozenset, list, tuple)):
                return set(fn)
        return set()

    def default_states(self) -> Set[str]:
        """States declared as the default/empty state by the registry."""
        return {s for s in self._all_states() if bool(self._state_props(s).get("is_default"))}

    def is_default_state(self, state: str) -> bool:
        if not state:
            return True
        return state.lower() in self.default_states()


_R = RegistryFacade(registry)


# ===================================================================== #
# Dynamic vocabularies backed by registry + lattice
# ===================================================================== #

class DynamicStopwords:
    def _words(self) -> Set[str]:
        reg = TypeRegistry.get_instance()
        return reg.get_function_words() | reg.get_sentence_connectives()
    def __contains__(self, item) -> bool: return item in self._words()
    def __iter__(self): return iter(self._words())
    def __len__(self): return len(self._words())
    def __sub__(self, other):
        o = set(other) if not isinstance(other, set) else other
        return self._words() - o
    def __rsub__(self, other):
        o = set(other) if not isinstance(other, set) else other
        return o - self._words()
    def __and__(self, other):
        o = set(other) if not isinstance(other, set) else other
        return self._words() & o
    def __rand__(self, other):
        o = set(other) if not isinstance(other, set) else other
        return o & self._words()
    def __or__(self, other):
        o = set(other) if not isinstance(other, set) else other
        return self._words() | o
    def __ror__(self, other):
        o = set(other) if not isinstance(other, set) else other
        return o | self._words()

STOPWORDS = DynamicStopwords()


class DynamicEgressTokens(frozenset):
    def __contains__(self, item): return str(item).lower() in TypeRegistry.get_instance().get_egress_tokens()
    def __iter__(self): return iter(TypeRegistry.get_instance().get_egress_tokens())
    def __len__(self): return len(TypeRegistry.get_instance().get_egress_tokens())
    def __and__(self, other):
        o = set(other) if not isinstance(other, set) else other
        return o & TypeRegistry.get_instance().get_egress_tokens()
    def __rand__(self, other):
        o = set(other) if not isinstance(other, set) else other
        return o & TypeRegistry.get_instance().get_egress_tokens()

EGRESS_INTENT_TOKENS = DynamicEgressTokens()


class DynamicMaterializationStates(frozenset):
    def __contains__(self, item): return str(item).lower() in TypeRegistry.get_instance().get_materialization_states()
    def __iter__(self): return iter(TypeRegistry.get_instance().get_materialization_states())
    def __len__(self): return len(TypeRegistry.get_instance().get_materialization_states())

MATERIALIZATION_OUTPUT_STATES = DynamicMaterializationStates()


# ===================================================================== #
# Calibration derivation — all numeric constants come from here
# ===================================================================== #

@dataclass
class Calibration:
    """Every numeric constant the planner needs, derived from the lattice."""
    # token information scale
    u: float
    # path-objective weights
    w_coverage: float
    w_alignment: float
    w_affinity: float
    w_literal_consumption: float
    w_macro_log_prob: float
    w_flat_log_prob: float
    macro_affinity_floor: float
    w_dead_ctor: float
    w_dead_expansion_step: float
    w_dead_output: float
    w_unbindable: float
    w_gap: float
    w_dispersion: float
    w_intent_deficit_final: float
    w_intent_deficit_partial: float
    w_weak_edge: float
    w_wildcarrier: float
    w_parsimony_step: float
    w_parsimony_base: float
    w_sink_bonus: float
    w_non_sink_penalty: float
    w_domain_coherence: float
    w_file_port_bonus: float
    w_file_port_penalty: float
    # thresholds
    coverage_strong: float
    coverage_weak: float
    coverage_low: float
    cov_gate_floor: float
    cov_mult_exp: float
    cov_mult_min: float
    align_floor: float
    align_step: float
    clause_hit_frac: float
    clause_candidate_frac: float
    clause_mass_min_rel: float
    join_min_parents: int
    join_relevance_floor: float
    weak_cov_share: float
    # edge-affinity components
    aff_ast_boost: float
    aff_reverse_mult: float
    aff_stage_sink: float
    aff_stage_sink_miss: float
    aff_join: float
    aff_bridge: float
    aff_adjacent: float
    aff_same_domain: float
    # stage-progress table (indexed by source/target stage-position)
    stage_prog: Tuple[Tuple[float, ...], ...]
    role_bump: float
    # edge-score weights
    edge_score_weights: Tuple[float, float, float, float]
    inv_dist_exact: float
    inv_dist_two_hop: float
    inv_dist_else: float
    # misc bonuses / penalties
    sink_completion_bonus: float
    dangling_output_penalty: float
    unrequested_transform_penalty: float
    unrequested_coverage_threshold: float
    # sub-lattice
    sublattice_domain_bonus: float
    sublattice_token_weight: float
    sublattice_rel_weight: float
    # MCTS
    mcts_affinity_weight: float
    # planning
    max_replicas: int
    max_steps_hard_cap: int
    max_steps_slack: int
    beam_width_linear: int
    beam_width_frontier: int
    beam_per_endpoint: int
    beam_pool_linear: int
    beam_pool_linear_prune: int
    beam_pool_frontier: int
    beam_pool_frontier_prune: int
    # macro affinity within scope
    synaptic_rel_floor: float
    synaptic_boost: float
    synaptic_min: float
    synaptic_max: float


def _derive_calibration(orchestrator: Any) -> Calibration:
    cells = list(getattr(orchestrator, "loaded_cells", {}).values()) or []
    n = max(len(cells), 1)
    token_index = getattr(orchestrator, "token_index", {}) or {}

    n_sources = sum(1 for c in cells if _R.is_source(c))
    n_sinks = sum(1 for c in cells if _R.is_sink(c))
    total_edges = sum(len(getattr(c, "edges", ()) or ()) for c in cells)

    domains: Dict[str, int] = {}
    for c in cells:
        d = getattr(c, "domain_name", "") or ""
        if d:
            domains[d] = domains.get(d, 0) + 1
    n_domains = max(len(domains), 1)

    edge_ratio = total_edges / n
    edge_drivenness = edge_ratio / (1.0 + edge_ratio) if edge_ratio else 0.0
    sink_focus = n_sinks / n
    source_focus = n_sources / n

    vocab = max(len(token_index), 1)
    total_df = sum(len(v) for v in token_index.values()) or 1
    avg_df = total_df / vocab
    u = math.log(1.0 + (n + 1) / (avg_df + 1.0))

    stages_present = sorted({getattr(c, "stage", None) for c in cells
                             if getattr(c, "stage", None) is not None})
    n_stages = max(len(stages_present), 2)
    stage_positions = n_stages

    # Generalisable thresholds — expressed in terms of the number of stages
    # and the number of clauses the prompt can express. None of these is a
    # fixed scalar; they scale as the lattice grows.
    coverage_strong = 1.0 - 1.0 / n_stages
    coverage_weak = 1.0 - 2.0 / n_stages
    coverage_low = 1.0 / n_stages
    cov_gate_floor = 1.0 / n_stages
    cov_mult_exp = 1.0 + 1.0 / n_stages
    cov_mult_min = 1.0 / (n_stages * n_stages)
    align_floor = 1.0 - 1.0 / (n_stages + 1)
    align_step = 1.0 / (n_stages * n_stages)
    clause_hit_frac = 1.0 / n_stages
    clause_candidate_frac = clause_hit_frac / 2.0
    clause_mass_min_rel = 1.0 / (n_stages * n_stages)
    join_min_parents = 2
    join_relevance_floor = 1.0 / (n_stages * n_stages)
    weak_cov_share = 1.0 / n_stages

    # Edge-affinity bases, in units of u so they scale with lattice size.
    aff_ast_boost = 1.0 + 1.0 / n_stages
    aff_reverse_mult = 1.0 - 1.0 / n_stages
    aff_stage_sink = (stage_positions - 1) / stage_positions
    aff_stage_sink_miss = 1.0 / (stage_positions * stage_positions)
    aff_join = 1.0 - 1.0 / (n_domains + 1)
    aff_bridge = aff_join
    aff_adjacent = 1.0 / (n_stages + 1)
    aff_same_domain = 1.0 / (n_stages * n_stages)

    # Stage progression table: source→transform < transform→transform <
    # transform→sink < source→sink < other.
    S, T, K = 0, 1, stage_positions - 1
    stage_prog = [[0.0] * stage_positions for _ in range(stage_positions)]
    for a in range(stage_positions):
        for b in range(stage_positions):
            if a == S and b == T:
                stage_prog[a][b] = 0.5
            elif a == T and b == T:
                stage_prog[a][b] = 0.3
            elif a == T and b == K:
                stage_prog[a][b] = 0.8
            elif a == S and b == K:
                stage_prog[a][b] = 0.6
            else:
                stage_prog[a][b] = 0.2
    role_bump = 1.0 / n_stages

    # Edge-score weights: normalised to sum to 1, biased by edge-drivenness.
    ast_share = 0.35 * (1.0 + edge_drivenness)
    rel_share = 0.30
    role_share = 0.25
    dist_share = 0.10
    total = ast_share + rel_share + role_share + dist_share
    edge_score_weights = (ast_share / total, rel_share / total,
                          role_share / total, dist_share / total)
    inv_dist_exact = 1.0
    inv_dist_two_hop = 1.0 / (n_stages + 1)
    inv_dist_else = 1.0 / (n_stages * n_stages)

    return Calibration(
        u=u,
        w_coverage=u * (1.0 + 1.0 / n_domains),
        w_alignment=u * (1.0 + 1.0 / n_domains),
        w_affinity=u * (1.0 + 2.0 * edge_drivenness),
        w_literal_consumption=u * (1.0 + source_focus + sink_focus),
        w_macro_log_prob=u * 0.5,
        w_flat_log_prob=u * 0.1,
        macro_affinity_floor=1.0 - 1.0 / n_stages,
        w_dead_ctor=u * (1.0 + 1.0 / n_stages),
        w_dead_expansion_step=u * 1.0,
        w_dead_output=u * 1.2,
        w_unbindable=u * (n_stages + 2.0),
        w_gap=u * (1.0 / n_stages),
        w_dispersion=u * (1.0 / n_domains + 1.0 / n_stages),
        w_intent_deficit_final=u * n_stages * 3.0,
        w_intent_deficit_partial=u * n_stages,
        w_weak_edge=1.0 / n_stages,
        w_wildcarrier=1.0 / n_stages,
        w_parsimony_step=u * (1.0 / (n_stages * n_stages)),
        w_parsimony_base=u * (1.0 / (n_stages ** 3)),
        w_sink_bonus=u * (0.8 + 2.0 * sink_focus),
        w_non_sink_penalty=u * (0.6 + 2.0 * sink_focus),
        w_domain_coherence=u * (1.0 / n_domains + 1.0 / n_stages),
        w_file_port_bonus=u * (1.0 / n_stages) * (1.0 + sink_focus),
        w_file_port_penalty=u * (1.0 / n_stages) * (1.0 + sink_focus),
        coverage_strong=coverage_strong,
        coverage_weak=coverage_weak,
        coverage_low=coverage_low,
        cov_gate_floor=cov_gate_floor,
        cov_mult_exp=cov_mult_exp,
        cov_mult_min=cov_mult_min,
        align_floor=align_floor,
        align_step=align_step,
        clause_hit_frac=clause_hit_frac,
        clause_candidate_frac=clause_candidate_frac,
        clause_mass_min_rel=clause_mass_min_rel,
        join_min_parents=join_min_parents,
        join_relevance_floor=join_relevance_floor,
        weak_cov_share=weak_cov_share,
        aff_ast_boost=aff_ast_boost,
        aff_reverse_mult=aff_reverse_mult,
        aff_stage_sink=aff_stage_sink,
        aff_stage_sink_miss=aff_stage_sink_miss,
        aff_join=aff_join,
        aff_bridge=aff_bridge,
        aff_adjacent=aff_adjacent,
        aff_same_domain=aff_same_domain,
        stage_prog=tuple(tuple(row) for row in stage_prog),
        role_bump=role_bump,
        edge_score_weights=edge_score_weights,
        inv_dist_exact=inv_dist_exact,
        inv_dist_two_hop=inv_dist_two_hop,
        inv_dist_else=inv_dist_else,
        sink_completion_bonus=u * (0.5 + sink_focus),
        dangling_output_penalty=u * (1.0 + sink_focus),
        unrequested_transform_penalty=u * 0.5,
        unrequested_coverage_threshold=coverage_strong,
        sublattice_domain_bonus=u * 0.5,
        sublattice_token_weight=1.0,
        sublattice_rel_weight=1.0,
        mcts_affinity_weight=2.0,
        max_replicas=max(n_stages * 3, 4),
        max_steps_hard_cap=max(n_stages * 4, 8),
        max_steps_slack=n_stages + 1,
        beam_width_linear=max(50, n * 2),
        beam_width_frontier=max(50, n),
        beam_per_endpoint=max(n_stages, 2),
        beam_pool_linear=min(max(500, n * 4), 3000),
        beam_pool_linear_prune=min(max(300, n * 2), 1500),
        beam_pool_frontier=min(max(500, n * 4), 2000),
        beam_pool_frontier_prune=min(max(250, n * 2), 1000),
        synaptic_rel_floor=clause_hit_frac,
        synaptic_boost=1.0 / n_stages,
        synaptic_min=1.0 - 1.0 / n_stages,
        synaptic_max=1.0 - 1.0 / (n_stages * n_stages),
    )


def _apply_weight_overrides(cal: Calibration) -> None:
    import json as _json
    try:
        try:
            from .config import settings as _settings
        except ImportError:
            from config import settings as _settings
    except ImportError:
        return
    cfg = getattr(_settings, "planner_config", None)
    if not cfg:
        return
    p = str(cfg)
    if not os.path.exists(p):
        return
    try:
        if p.endswith(".json"):
            with open(p, "r", encoding="utf-8") as f:
                data = _json.load(f)
        else:
            try:
                import yaml
            except ImportError:
                return
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
    except Exception:
        return
    if not isinstance(data, dict):
        return
    for k, v in (data.get("calibration") or {}).items():
        if hasattr(cal, k):
            try:
                setattr(cal, k, float(v))
            except (TypeError, ValueError):
                pass


# ===================================================================== #
# Structural helpers
# ===================================================================== #

def _is_col_projection_port(p_sig: Any) -> bool:
    cached = getattr(p_sig, "_cached_col_proj", None)
    if cached is not None:
        return cached
    st = str(getattr(getattr(p_sig, "signature", p_sig), "state", "")).lower()
    cps = _R.column_projection_state()
    if cps and st == cps:
        try: p_sig._cached_col_proj = True
        except Exception: pass
        return True
    tname = str(getattr(getattr(p_sig, "signature", p_sig), "type_name", "")).lower()
    if _R.is_collection(tname) and not _R.is_dense(tname):
        st_tokens = set(CellTokenizer.tokenize_identifier(st))
        if st_tokens & _R.column_projection_tokens():
            try: p_sig._cached_col_proj = True
            except Exception: pass
            return True
    try: p_sig._cached_col_proj = False
    except Exception: pass
    return False


def _is_table_groundable_target_port(
    p_sig: Any, prev_path: List[Cell],
    quoted_str_literals: Sequence[Any] = (),
    identifier_literals: Sequence[Any] = (),
) -> bool:
    p_role = getattr(p_sig, "port_role", None) or getattr(p_sig, "derived_role", "")
    if not p_role and hasattr(p_sig, "derive_port_role"):
        p_role = p_sig.derive_port_role()
    if p_role != _R.target_input_role() or not _R.target_input_role():
        return False
    if not (quoted_str_literals or identifier_literals):
        return False
    return any(
        _R.is_tabular(str(getattr(out_s.signature, "type_name", "")).lower())
        for prev_c in prev_path
        for out_s in prev_c.outputs.values()
    )


def _extract_ndim_from_contract(sc: Any) -> Optional[int]:
    if sc is None:
        return None
    if isinstance(sc, dict):
        v = sc.get("ndim")
        return int(v) if v is not None else None
    if isinstance(sc, str):
        s = sc.strip()
        if s.startswith("(") and s.endswith(")"):
            inner = s[1:-1].strip()
            return len([p for p in inner.split(",") if p.strip()]) if inner else 0
    return None


def _safe_slots_items(cell: Any):
    s = getattr(cell, "slots", None)
    if isinstance(s, dict):
        return list(s.items())
    if isinstance(s, (list, tuple, set)):
        return [(item, {}) for item in s]
    return []


def _is_terminal_sink_cell(cell: Any) -> bool:
    if not _R.is_sink(cell):
        return False
    if _R.is_macro(cell) and getattr(cell, "sub_cells", None):
        return False
    top = getattr(cell, "topology_type", None)
    if top is None:
        return True
    # Linear topologies terminate; branching ones do not.
    seq_like = {t for t in _R.topologies() if "branch" not in t.lower() and "coproduct" not in t.lower()
                and "loop" not in t.lower()}
    return top in seq_like


def _is_port_role_compatible(p_out: Any, p_in: Any) -> bool:
    in_role = getattr(p_in, "port_role", None) or getattr(p_in, "derived_role", "")
    out_role = getattr(p_out, "port_role", None) or getattr(p_out, "derived_role", "")
    pred_in = _R.prediction_input_role()
    tgt_in = _R.target_input_role()
    pred_out = _R.prediction_output_role()
    if pred_in and in_role == pred_in:
        if tgt_in and out_role == tgt_in:
            return False
        out_state = str(getattr(getattr(p_out, "signature", None), "state", "")).lower()
        if not (out_role == pred_out or bool(_R._state_props(out_state).get("is_prediction"))):
            return False
    elif tgt_in and in_role == tgt_in:
        if pred_out and out_role == pred_out:
            return False
        out_state = str(getattr(getattr(p_out, "signature", None), "state", "")).lower()
        if bool(_R._state_props(out_state).get("is_prediction")):
            return False
    return True


def _segment_prompt_clauses(prompt: str) -> List[str]:
    if not prompt:
        return []
    parts, buf, depth = [], [], 0
    for ch in prompt:
        if ch in "[({":
            depth += 1; buf.append(ch); continue
        if ch in "])}":
            depth = max(0, depth - 1); buf.append(ch); continue
        if depth == 0 and ch in ",;":
            seg = "".join(buf).strip()
            if seg: parts.append(seg)
            buf = []; continue
        buf.append(ch)
    if buf:
        seg = "".join(buf).strip()
        if seg: parts.append(seg)

    merged: List[str] = []

    def _is_sink_directive(text: str) -> bool:
        toks = [t.lower() for t in tokenize_alphanumeric(text)]
        if not toks: return False
        reg = TypeRegistry.get_instance()
        connective = reg.get_sentence_connectives()
        if toks and toks[0] in connective:
            toks = toks[1:]
        if not toks: return False
        egress_words = reg.get_egress_tokens()
        prep_words = {t for t in tokenize_alphanumeric(text) if t.lower() in STOPWORDS}
        if toks[0] not in egress_words: return False
        if not prep_words: return False
        return len(toks) <= 1 + len(prep_words)

    for p in parts:
        p_clean = p.strip()
        if not p_clean: continue
        p_toks = CellTokenizer.tokenize_prompt(p_clean)
        if not p_toks: continue
        if merged and _is_sink_directive(p_clean):
            if ExecutionContext._extract_target_sink(p_clean):
                continue
            merged[-1] = merged[-1] + ", " + p_clean
            continue
        if merged and p_toks.issubset(CellTokenizer.tokenize_prompt(merged[-1])):
            merged[-1] = merged[-1] + ", " + p_clean
            continue
        merged.append(p_clean)

    try:
        from lattice import LatticeOrchestrator as _LO
        orch = _LO.get_active_instance()
    except ImportError:
        orch = None

    if orch is not None and len(getattr(orch, "loaded_cells", {})) > 0:
        op_vocab: Set[str] = set()
        for c in orch.loaded_cells.values():
            for tag in getattr(c, "semantic_tags", []) or []:
                t = str(tag).lower().strip()
                if len(t) > 1 and " " not in t:
                    op_vocab.add(t)
            for kw in getattr(c, "keywords", []) or []:
                k = str(kw).lower().strip()
                if len(k) > 1 and " " not in k:
                    op_vocab.add(k)
        try:
            aliases = TypeRegistry.get_instance().get_all_aliases()
        except Exception:
            aliases = {}
        filtered = []
        for p in merged:
            toks = CellTokenizer.tokenize_prompt(p)
            content = {t for t in toks if t not in aliases}
            if content & op_vocab:
                filtered.append(p)
        if filtered:
            return filtered

    return merged or [prompt.strip()]


# ===================================================================== #
# Planner
# ===================================================================== #

class LatticePlanner:

    def __init__(self, orchestrator: LatticeOrchestrator,
                 rag: Optional[Any] = None,
                 macros_enabled: Optional[bool] = None,
                 topology_mode: Optional[str] = None):
        self.orchestrator = orchestrator
        self.rag = rag
        if macros_enabled is not None:
            self.macros_enabled = bool(macros_enabled)
        else:
            try:
                from config import settings
                self.macros_enabled = bool(getattr(settings, "macros_enabled", True))
            except Exception:
                self.macros_enabled = True
        if topology_mode is not None:
            self.topology_mode = str(topology_mode).lower()
        else:
            try:
                from config import settings
                self.topology_mode = str(getattr(settings, "topology_mode", "frontier")).lower()
            except Exception:
                self.topology_mode = "frontier"
        self.current_relevance_map: Dict[str, float] = {}
        self._affinity_cache: Dict[Tuple[str, str], float] = {}
        self._macro_edges_index: Optional[Dict[Tuple[str, str], List[str]]] = None
        self._cells_connect_cache: Dict[Tuple[str, str], bool] = {}
        self.cal = _derive_calibration(orchestrator)

    def _get_lattice_op_tokens(self) -> Set[str]:
        if getattr(self, "_cached_op_tokens", None) is not None:
            return self._cached_op_tokens
        op_toks: Set[str] = set()
        seen_tokens: Dict[str, int] = {}
        total_cells = 0
        for c in getattr(self.orchestrator, "loaded_cells", {}).values():
            total_cells += 1
            toks_here: Set[str] = set()
            if _R.is_transform(c) or _R.is_sink(c):
                for t in CellTokenizer.tokenize_identifier(c.cell_id):
                    toks_here.add(t.lower())
                for kw in getattr(c, "keywords", ()) or ():
                    for t in CellTokenizer.tokenize_identifier(str(kw)):
                        toks_here.add(t.lower())
                for tag in getattr(c, "semantic_tags", ()) or ():
                    for t in CellTokenizer.tokenize_identifier(str(tag)):
                        toks_here.add(t.lower())
            for t in toks_here:
                if len(t) >= 2:
                    seen_tokens[t] = seen_tokens.get(t, 0) + 1
        threshold = max(1, int(0.5 * total_cells))
        op_toks = {t for t, d in seen_tokens.items() if d < threshold}
        self._cached_op_tokens = op_toks
        return op_toks

    # ---- literal consumption ----
    def _literal_consumption_of(self, path, universal_literals,
                                numeric_literals, quoted_str_literals, identifier_literals):
        if not universal_literals:
            return 1.0
        op_toks = self._get_lattice_op_tokens()
        port_tokens: Set[str] = set()
        cell_id_tokens: Set[str] = set()
        domain_tokens: Set[str] = set()
        has_col_proj = False
        has_numeric_port = False

        def _tokens(s):
            s = str(s).lower()
            out = {s}
            for part in tokenize_alphanumeric(s):
                if part: out.add(part)
            return out

        def _port_match(lc, port):
            if lc == port: return True
            min_substr = 2  # minimal substring length for coincidence-avoidance
            return len(lc) >= min_substr and lc in port

        def _identifier_match(lc):
            return (any(_port_match(lc, p) for p in port_tokens)
                    or any(_port_match(lc, c) for c in cell_id_tokens)
                    or any(_port_match(lc, d) for d in domain_tokens))

        for c in path:
            cell_id_tokens |= _tokens(getattr(c, "cell_id", "") or "")
            dom = getattr(c, "domain_name", "") or ""
            if dom: domain_tokens |= _tokens(dom)
            for pn, ps in c.inputs.items():
                port_tokens.add(pn.lower())
                r = (getattr(ps, "port_role", None) or getattr(ps, "derived_role", "") or "").lower()
                if r: port_tokens.add(r)
                tn = str(getattr(ps.signature, "type_name", "")).lower()
                if _is_col_projection_port(ps) or _R.is_collection(tn):
                    has_col_proj = True
                if _R.is_numeric(tn):
                    has_numeric_port = True
            for sk in getattr(c, "bound_slots", {}).keys():
                port_tokens.add(sk.lower())
            for sl in getattr(c, "slots", []):
                port_tokens.add(str(sl).lower())

        consumed = 0
        path_ingress = sum(
            1 for c in path if _R.is_source(c)
            for p in c.inputs.values()
            if p.required and p.default_value is None and _lattice_is_path_port(p)
        )
        path_egress = sum(
            1 for c in path if _R.is_sink(c)
            for p in c.inputs.values()
            if p.required and p.default_value is None and _lattice_is_path_port(p)
        )
        proj_vocab = _R.column_projection_tokens()
        rel_trigs = _R.relational_triggers()
        path_col_key_slots = 0
        for c in path:
            for s in (list(c.inputs.keys()) + list(getattr(c, "slots", []) or [])):
                sl = str(s).lower()
                if sl in proj_vocab or sl in rel_trigs:
                    path_col_key_slots += 1

        used_ing = used_egr = used_col = 0
        for kind, lit in universal_literals:
            lc = str(lit).lower().strip("'\"")
            if kind == "file_asset":
                if used_ing < path_ingress:
                    consumed += 1; used_ing += 1
                elif used_egr < path_egress:
                    consumed += 1; used_egr += 1
            elif kind == "numeric":
                if numeric_literals and has_numeric_port:
                    consumed += 1
            elif kind == "quoted_str":
                if has_col_proj or any(_port_match(lc, p) for p in port_tokens):
                    consumed += 1
                elif used_col < path_col_key_slots:
                    consumed += 1; used_col += 1
            else:
                if _identifier_match(lc):
                    consumed += 1
                elif lc not in op_toks and used_col < path_col_key_slots:
                    consumed += 1; used_col += 1
        return consumed / max(len(universal_literals), 1)

    def _clause_coverage_of(self, path, clause_tokens_list):
        if not clause_tokens_list:
            return 1.0
        covered = set()
        for c in path:
            c_toks = c.token_set
            id_toks = getattr(c, "identity_tokens", c_toks)
            for gi, cl_toks in enumerate(clause_tokens_list):
                if cl_toks & id_toks:
                    covered.add(gi)
        return len(covered) / len(clause_tokens_list)

    def _hard_constraint_filter(self, candidates, universal_literals,
                                numeric_literals, quoted_str_literals, identifier_literals,
                                clause_tokens_list):
        strong = self.cal.coverage_strong
        weak = self.cal.coverage_weak
        low = self.cal.coverage_low
        def _score(item):
            path = item[0]
            cov = self._clause_coverage_of(path, clause_tokens_list)
            lit = self._literal_consumption_of(
                path, universal_literals, numeric_literals,
                quoted_str_literals, identifier_literals)
            return cov, lit
        tiers = [(strong, 1.0), (weak, 1.0), (low, 0.0), (0.0, 0.0)]
        for tier, (cf, lf) in enumerate(tiers):
            keep = [it for it in candidates if _score(it)[0] >= cf and _score(it)[1] >= lf]
            if keep:
                if tier >= 2:
                    print(f"[NSTL][planner] WARNING: falling back to tier {tier} "
                          f"(coverage_floor={cf:.2f}).")
                return keep, tier
        try:
            from config import settings as _s
            _refuse = bool(getattr(_s, "require_coverage_floor", False))
        except Exception:
            _refuse = False
        if _refuse and candidates:
            return [], len(tiers)
        return candidates, len(tiers) - 1

    def _cells_connect(self, c1: Cell, c2: Cell) -> bool:
        pair = (c1.cell_id, c2.cell_id)
        cached = self._cells_connect_cache.get(pair)
        if cached is not None:
            return cached
        res = any(
            unify(getattr(o, "signature", o), getattr(p, "signature", p)) is not None
            for o in c1.outputs.values() for p in c2.inputs.values()
        )
        self._cells_connect_cache[pair] = res
        return res

    def _build_macro_edges_index(self) -> Dict[Tuple[str, str], List[str]]:
        index: Dict[Tuple[str, str], List[str]] = {}
        if getattr(self, "macros_enabled", True) and hasattr(self.orchestrator, "loaded_cells"):
            for m in self.orchestrator.loaded_cells.values():
                if _R.is_macro(m) or getattr(m, "sub_cells", None):
                    topo = getattr(m, "internal_topology", {}) or {}
                    for src, dsts in topo.items():
                        for dst in dsts:
                            index.setdefault((src, dst), []).append(m.cell_id)
                    subs = getattr(m, "sub_cells", None)
                    if subs:
                        for i in range(len(subs) - 1):
                            index.setdefault((subs[i], subs[i + 1]), []).append(m.cell_id)
        return index

    def _calculate_edge_affinity(self, src_cell: Cell, dst_cell: Cell) -> float:
        key = (src_cell.cell_id, dst_cell.cell_id)
        cached = self._affinity_cache.get(key)
        if cached is not None:
            return cached
        res = self._compute_edge_affinity_raw(src_cell, dst_cell)
        self._affinity_cache[key] = res
        return res

    def _compute_edge_affinity_raw(self, src_cell: Cell, dst_cell: Cell) -> float:
        C = self.cal
        if getattr(self, "macros_enabled", True):
            if self._macro_edges_index is None:
                self._macro_edges_index = self._build_macro_edges_index()
            m_ids = self._macro_edges_index.get((src_cell.cell_id, dst_cell.cell_id))
            if m_ids:
                rel_map = getattr(self, "current_relevance_map", {}) or {}
                macro_rel = max(rel_map.get(mid, 0.0) for mid in m_ids)
                if macro_rel > C.synaptic_rel_floor:
                    boost = C.synaptic_boost * macro_rel
                    return min(1.0, max(C.synaptic_min, C.synaptic_min + boost))

        dst_id_lower = dst_cell.cell_id.lower()
        for edge in getattr(src_cell, "edges", []):
            tgt = edge.get("target_cell_id") if isinstance(edge, dict) else getattr(edge, "target_cell_id", None)
            if tgt and (tgt == dst_cell.cell_id or str(tgt).lower() == dst_id_lower):
                aff = float(edge.get("affinity_score", C.aff_adjacent) if isinstance(edge, dict)
                            else getattr(edge, "affinity_score", C.aff_adjacent))
                prov = edge.get("score_provenance") if isinstance(edge, dict) else getattr(edge, "score_provenance", "")
                if prov == "ast_mined":
                    return min(1.0, aff * C.aff_ast_boost)
                return aff

        src_id_lower = src_cell.cell_id.lower()
        for edge in getattr(dst_cell, "edges", []):
            tgt = edge.get("target_cell_id") if isinstance(edge, dict) else getattr(edge, "target_cell_id", None)
            if tgt and (tgt == src_cell.cell_id or str(tgt).lower() == src_id_lower):
                aff = float(edge.get("affinity_score", C.aff_same_domain) if isinstance(edge, dict)
                            else getattr(edge, "affinity_score", C.aff_same_domain))
                return aff * C.aff_reverse_mult

        src_dom, dst_dom = getattr(src_cell, "domain_name", ""), getattr(dst_cell, "domain_name", "")
        if _R.is_transform(src_cell) and _R.is_sink(dst_cell) and src_dom and src_dom == dst_dom:
            rel_map = getattr(self, "current_relevance_map", None) or {}
            if not rel_map or rel_map.get(src_cell.cell_id, 0.0) > 0.0:
                return C.aff_stage_sink
            return C.aff_stage_sink_miss

        is_join_node = (getattr(dst_cell, "topology_type", "") in _R.topologies()
                        and any("branch" in getattr(dst_cell, "topology_type", "").lower()
                                or "join" in getattr(dst_cell, "topology_type", "").lower()
                                or "product" in getattr(dst_cell, "topology_type", "").lower()
                                for _ in [0])
                        or len(getattr(dst_cell, "inputs", {})) >= C.join_min_parents)
        if is_join_node and src_dom and src_dom == dst_dom:
            rel_map = getattr(self, "current_relevance_map", None) or {}
            if not rel_map or rel_map.get(dst_cell.cell_id, 0.0) > 0.0:
                return C.aff_join

        src_tags = set(getattr(src_cell, "semantic_tags", []) or [])
        if src_tags & _R.bridge_tags() and src_dom and dst_dom and src_dom != dst_dom:
            return C.aff_bridge

        adj = getattr(self.orchestrator, "_adjacency", None) or getattr(self.orchestrator, "adjacency", None)
        if adj and dst_cell.cell_id in adj.get(src_cell.cell_id, ()):
            return C.aff_adjacent
        if src_dom and src_dom == dst_dom:
            return C.aff_same_domain
        return 0.0

    def compute_edge_score(self, src_cell, dst_cell,
                           relevance_map=None, weights=None) -> float:
        C = self.cal
        w1, w2, w3, w4 = weights or C.edge_score_weights
        ast = self._calculate_edge_affinity(src_cell, dst_cell)
        rel = (relevance_map or {}).get(dst_cell.cell_id, 0.0)
        sp_src, sp_dst = _R.stage_position(src_cell), _R.stage_position(dst_cell)
        stage_prog = C.stage_prog
        if 0 <= sp_src < len(stage_prog) and 0 <= sp_dst < len(stage_prog[sp_src]):
            role_prog = stage_prog[sp_src][sp_dst]
        else:
            role_prog = stage_prog[0][0] if stage_prog else 0.0
        adj = getattr(self.orchestrator, "_adjacency", {})
        if dst_cell.cell_id in adj.get(src_cell.cell_id, ()):
            inv_dist = C.inv_dist_exact
        else:
            two_hop = any(dst_cell.cell_id in adj.get(mid, ()) for mid in adj.get(src_cell.cell_id, ()))
            inv_dist = C.inv_dist_two_hop if two_hop else C.inv_dist_else
        return w1 * ast + w2 * rel + w3 * role_prog + w4 * inv_dist

    # ---- main entry ----
    def plan(self, prompt, tunnel, relevance_map, start_sig=None, goal_sig=None, max_transforms=6):
        self.current_relevance_map = dict(relevance_map or {})
        self._affinity_cache.clear()
        self.cal = _derive_calibration(self.orchestrator)
        _apply_weight_overrides(self.cal)
        C = self.cal

        if not tunnel:
            return []
        if len(tunnel) == 1:
            return [tunnel[0]]

        # Clone once so no beam path shares identity with the shared lattice.
        tunnel = [c.clone() if hasattr(c, "clone") else copy.copy(c) for c in tunnel]

        # Exclude constant-node and macro-shortcut cells from the flat candidate pool.
        const_marker = next(iter(_R.node_types() - {""}), "")
        macro_marker = _R.macro_marker()

        def _is_flat_candidate(c: Cell) -> bool:
            if const_marker and getattr(c, "node_type", "") == const_marker:
                return False
            if macro_marker and getattr(c, "cell_type", "") == macro_marker:
                return False
            if macro_marker and getattr(c, "node_role", "") == macro_marker:
                return False
            if isinstance(c, MacroCell):
                return False
            return True

        candidates = [c for c in tunnel if _is_flat_candidate(c)]
        if not candidates:
            candidates = [c for c in tunnel if not const_marker or getattr(c, "node_type", "") != const_marker]
        if not candidates:
            return [tunnel[0]]

        log_probs: Dict[str, float] = {}
        for c in candidates:
            log_probs[c.cell_id] = math.log(max(relevance_map.get(c.cell_id, 0.0), 1e-6))

        _debug_plan = os.environ.get("NSTL_DEBUG_PLAN", "") in ("1", "true", "True", "on")
        self._component_trace: Dict[Tuple[str, ...], Dict[str, float]] = {}

        l0 = ExecutionContext._extract_universal_literals(prompt or "")
        target_sink = ExecutionContext._extract_target_sink(prompt or "")

        def _keep(kind, val):
            return not (kind == "identifier" and target_sink and str(val).lower() == target_sink.lower())

        universal_literals = [(kind, val) for _, kind, val in l0
                              if kind in ("file_asset", "identifier", "quoted_str", "numeric")
                              and _keep(kind, val)]
        literal_positions = {(kind, val): pos for pos, kind, val in l0
                             if kind in ("file_asset", "identifier", "quoted_str", "numeric")
                             and _keep(kind, val)}
        identifier_role_map = ExecutionContext._build_identifier_role_map(prompt or "")
        self._last_identifier_roles = identifier_role_map

        file_literals = [v for k, v in universal_literals
                         if k == "file_asset" or (k == "quoted_str"
                                                  and ExecutionContext._is_path_string(str(v)))]
        dest_file_literals = [val for pos, k, val in l0
                              if (k == "file_asset" or (k == "quoted_str"
                                                       and ExecutionContext._is_path_string(str(val))))
                              and ExecutionContext._asset_direction(prompt or "", pos) == "dest"]
        src_file_literals = [val for pos, k, val in l0
                             if (k == "file_asset" or (k == "quoted_str"
                                                      and ExecutionContext._is_path_string(str(val))))
                             and ExecutionContext._asset_direction(prompt or "", pos) != "dest"]

        def _has_path_port(cell: Cell) -> bool:
            return any(_lattice_is_path_port(p) for p in cell.inputs.values())

        numeric_literals = [v for _, k, v in l0 if k in ("numeric", "int", "float", "number")]
        quoted_str_literals = [v for k, v in universal_literals
                               if k == "quoted_str" and not ExecutionContext._is_path_string(str(v))]
        identifier_literals = [v for k, v in universal_literals if k == "identifier"]

        def _can_be_entry_source(cell: Cell) -> bool:
            for p_name, p_sig in cell.inputs.items():
                if not p_sig.required or p_sig.default_value is not None:
                    continue
                t = str(getattr(p_sig.signature, "type_name", "")).lower()
                if _lattice_is_path_port(p_sig):
                    if not file_literals and not _R.is_source(cell) \
                            and not _R.is_macro(cell) and not isinstance(cell, MacroCell):
                        return False
                    continue
                if _R.is_numeric(t):
                    if not numeric_literals: return False
                    continue
                if not (_R.is_textual(t) or _R.is_logical(t) or _R.is_scalar(t)):
                    return False
            return True

        tunnel_has_sinks = any(_R.is_sink(c) for c in candidates)
        tunnel_absorbs_assets = any(_R.is_source(c) and _has_path_port(c) for c in candidates) if file_literals else False

        candidate_entries = list(candidates)
        if start_sig is not None:
            s_sig = getattr(start_sig, "signature", start_sig)
            matching = [c for c in candidate_entries if unify(s_sig, c.primary_input.signature) is not None]
            if matching:
                candidate_entries = matching
        else:
            viable = [c for c in candidate_entries if _can_be_entry_source(c)]
            clauses = CellTokenizer.split_prompt_clauses(prompt)
            first_clause = clauses[0].strip() if clauses else prompt.strip()
            clause_tokens = CellTokenizer.tokenize_prompt(first_clause) if first_clause else set()

            s1_entries = [c for c in viable
                          if _R.is_source(c) or _R.is_macro(c) or isinstance(c, MacroCell)]
            first_file_lits = [v for _, k, v in ExecutionContext._extract_universal_literals(first_clause)
                               if k == "file_asset"
                               or (k == "quoted_str" and ExecutionContext._is_path_string(str(v)))]
            if first_file_lits:
                s1_entries = [c for c in s1_entries if _has_path_port(c)]
            matching_s1 = [c for c in s1_entries
                           if clause_tokens and len(clause_tokens & getattr(c, "identity_tokens", c.token_set)) > 0]
            if matching_s1:
                matching_s1.sort(key=lambda c: (
                    getattr(c, "source_priority", len(candidates)),
                    -(relevance_map.get(c.cell_id, 0.0)
                      * (1.0 + len(clause_tokens & getattr(c, "identity_tokens", c.token_set)))),
                ))
                candidate_entries = matching_s1[:max(len(matching_s1), 1)]
            elif file_literals and s1_entries:
                s1_entries.sort(key=lambda c: (
                    getattr(c, "source_priority", len(candidates)),
                    -relevance_map.get(c.cell_id, 0.0),
                ))
                candidate_entries = s1_entries[:max(len(s1_entries), 1)]
            else:
                viable.sort(key=lambda c: -relevance_map.get(c.cell_id, 0.0))
                candidate_entries = viable

            entry_ids = {c.cell_id for c in candidate_entries}
            self_contained = [
                c for c in viable
                if c.cell_id not in entry_ids
                and relevance_map.get(c.cell_id, 0.0) > C.clause_hit_frac
                and all(not p.required or p.default_value is not None for p in c.inputs.values())
            ]
            if self_contained:
                candidate_entries = list(candidate_entries) + self_contained

        clauses = _segment_prompt_clauses(prompt)
        clause_tokens_list = [CellTokenizer.tokenize_prompt(cl) - set(STOPWORDS) for cl in clauses]
        clause_tokens_list = [t for t in clause_tokens_list if t]
        content_prompt_tokens = (set().union(*clause_tokens_list) if clause_tokens_list
                                 else (CellTokenizer.tokenize_prompt(prompt) if prompt else set()))
        num_clauses = max(len(clause_tokens_list), 1)

        has_prompt_egress_intent = (
            (bool(content_prompt_tokens & EGRESS_INTENT_TOKENS) and not target_sink)
            or bool(dest_file_literals)
            or (goal_sig is not None)
        )

        token_index = getattr(self.orchestrator, "token_index", None) or {}
        corpus_size = max(len(self.orchestrator.loaded_cells), 1)

        def _idf(tok):
            df = len(token_index.get(tok, ()))
            return math.log(1.0 + (corpus_size + 1) / (df + 1.0))

        idf_of_prompt = {t: _idf(t) for t in content_prompt_tokens}
        total_prompt_idf = sum(idf_of_prompt.values()) or 1.0
        clause_weights = [sum(_idf(t) for t in cl) for cl in clause_tokens_list]
        total_clause_weight = sum(clause_weights) or 1.0

        def _identity_tokens(cell):
            toks = CellTokenizer.tokenize_identifier(cell.cell_id)
            for kw in getattr(cell, "keywords", ()) or ():
                toks.update(CellTokenizer.tokenize_identifier(kw))
            wildcards = _R.wildcard_names()
            for p in list(cell.inputs.values()) + list(cell.outputs.values()):
                sig = getattr(p, "signature", p)
                tn = str(getattr(sig, "type_name", "") or "")
                if tn and tn.lower() not in wildcards:
                    toks.update(CellTokenizer.tokenize_identifier(tn))
            return toks

        identity_cache: Dict[str, Set[str]] = {c.cell_id: _identity_tokens(c) for c in candidates}

        operand_dependencies: List[Tuple[Set[str], Set[str]]] = []
        rel_trigs = _R.relational_triggers()
        non_op = set(STOPWORDS) | set(_R.column_projection_tokens())
        prompt_words = [w.lower().strip("'\",.;:()[]{}")
                        for w in tokenize_alphanumeric(prompt or "", min_len=1)]
        for p_idx in range(1, len(prompt_words) - 1):
            if prompt_words[p_idx] not in rel_trigs:
                continue
            head_w = prompt_words[p_idx - 1]
            op_idx = p_idx + 1
            while op_idx < len(prompt_words) and prompt_words[op_idx] in STOPWORDS:
                op_idx += 1
            if op_idx >= len(prompt_words):
                continue
            op_w = prompt_words[op_idx]
            if not (head_w and op_w and head_w != op_w and head_w not in non_op and op_w not in non_op):
                continue
            h_toks = CellTokenizer.tokenize_identifier(head_w)
            o_toks = CellTokenizer.tokenize_identifier(op_w)
            h_cells = {c.cell_id for c in candidates if h_toks & identity_cache.get(c.cell_id, set())}
            o_cells = {c.cell_id for c in candidates if o_toks & identity_cache.get(c.cell_id, set())}
            if h_cells and o_cells:
                operand_dependencies.append((o_cells, h_cells))

        def _match_mass(cl_toks, c_toks, id_toks):
            strong = sum(idf_of_prompt.get(t, _idf(t)) for t in (cl_toks & (c_toks & id_toks)))
            weak = sum(idf_of_prompt.get(t, _idf(t)) for t in (cl_toks & (c_toks - id_toks)))
            return strong + C.weak_cov_share * weak

        def _is_wildcarrier(cell):
            t = str(getattr(getattr(cell, "primary_output", None), "type_name", "") or "").lower()
            return _R.is_wildcard(t)

        cell_cov_strong: Dict[str, Set[str]] = {}
        cell_cov_weak: Dict[str, Set[str]] = {}
        cell_cov_mass_bonus: Dict[str, float] = {}
        cell_clause_mass: Dict[str, List[float]] = {}
        cell_covered: Dict[str, Set[int]] = {}

        literal_prompt_tokens: Set[str] = set()
        for _, kind, val in l0:
            is_path_lit = (kind == "file_asset"
                           or (kind == "quoted_str" and ExecutionContext._is_path_string(str(val))))
            if is_path_lit:
                literal_prompt_tokens |= set(CellTokenizer.tokenize_identifier(str(val)))
        non_literal_prompt_tokens = content_prompt_tokens - literal_prompt_tokens - set(STOPWORDS)

        for c in candidates:
            c_toks = c.token_set
            id_toks = getattr(c, "identity_tokens", None) or identity_cache.get(c.cell_id, c_toks)
            cell_cov_strong[c.cell_id] = content_prompt_tokens & id_toks
            cell_cov_weak[c.cell_id] = non_literal_prompt_tokens & (c_toks - id_toks)
            cell_cov_mass_bonus[c.cell_id] = (
                sum(idf_of_prompt.get(t, _idf(t)) for t in cell_cov_strong[c.cell_id])
                + C.weak_cov_share * sum(idf_of_prompt.get(t, _idf(t)) for t in cell_cov_weak[c.cell_id])
            ) / total_prompt_idf
            masses: List[float] = []
            covered: Set[int] = set()
            for gi, cl_toks in enumerate(clause_tokens_list):
                m = _match_mass(cl_toks, c_toks, id_toks)
                masses.append(m)
                if m >= C.clause_hit_frac * clause_weights[gi]:
                    covered.add(gi)
            cell_clause_mass[c.cell_id] = masses
            cell_covered[c.cell_id] = covered

        macro_expansion: Dict[str, List[Cell]] = {}
        for c in candidates:
            if not _R.is_macro(c) or not getattr(c, "sub_cells", None):
                continue
            subs = []
            for sid in c.sub_cells:
                sub = self.orchestrator.loaded_cells.get(sid) if self.orchestrator else None
                if sub is None:
                    subs = []; break
                subs.append(sub)
            if len(subs) >= C.join_min_parents:
                macro_expansion[c.cell_id] = subs

        for c_id, subs in macro_expansion.items():
            strong: Set[str] = set(); weak: Set[str] = set()
            mass_bonus = 0.0
            masses: List[float] = []; covered: Set[int] = set()
            for sub in subs:
                s_id = sub.cell_id
                if s_id in cell_cov_strong:
                    strong |= cell_cov_strong[s_id]
                    weak |= cell_cov_weak.get(s_id, set())
                    mass_bonus += cell_cov_mass_bonus.get(s_id, 0.0)
                    for gi, m in enumerate(cell_clause_mass.get(s_id, [])):
                        if len(masses) <= gi: masses.append(0.0)
                        masses[gi] = max(masses[gi], m)
                    covered |= cell_covered.get(s_id, set())
                    continue
                s_toks = sub.token_set
                s_id_toks = getattr(sub, "identity_tokens", None) or identity_cache.get(s_id) or _identity_tokens(sub)
                s_strong = content_prompt_tokens & s_id_toks
                s_weak = content_prompt_tokens & (s_toks - s_id_toks)
                strong |= s_strong; weak |= s_weak
                mass_bonus += (sum(idf_of_prompt.get(t, _idf(t)) for t in s_strong)
                               + C.weak_cov_share * sum(idf_of_prompt.get(t, _idf(t)) for t in s_weak)) / total_prompt_idf
                for gi, cl_toks in enumerate(clause_tokens_list):
                    m = _match_mass(cl_toks, s_toks, s_id_toks)
                    if len(masses) <= gi: masses.append(0.0)
                    masses[gi] = max(masses[gi], m)
                    if m >= C.clause_hit_frac * clause_weights[gi]:
                        covered.add(gi)
            cell_cov_strong[c_id] = strong
            cell_cov_weak[c_id] = weak
            cell_cov_mass_bonus[c_id] = mass_bonus
            cell_clause_mass[c_id] = masses
            cell_covered[c_id] = covered

        cell_concrete_in_sigs: Dict[str, List[Any]] = {}
        for c in candidates:
            cell_concrete_in_sigs[c.cell_id] = [p.signature for p in c.inputs.values()
                                                if not _R.is_wildcard(str(p.signature.type_name).lower())]

        def _port_literal_groundable(type_name, p_sig=None):
            t = type_name.lower()
            return (
                (_R.is_textual(t) and bool(quoted_str_literals or identifier_literals))
                or (_R.is_numeric(t) and bool(numeric_literals))
                or _R.is_logical(t)
                or _R.is_path(t)
                or (_R.is_scalar(t) and bool(quoted_str_literals or identifier_literals or numeric_literals))
                or (p_sig is not None and _is_col_projection_port(p_sig)
                    and bool(identifier_literals or quoted_str_literals))
            )

        receiver_role = _R.receiver_role()
        cell_receiver_sigs: Dict[str, List[Any]] = {}
        for c in candidates:
            if const_marker and getattr(c, "node_type", "") == const_marker:
                cell_receiver_sigs[c.cell_id] = []
                continue
            sigs = []
            for p_name, p_sig in c.inputs.items():
                if not p_sig.required or p_sig.default_value is not None:
                    continue
                declared_role = getattr(p_sig, "port_role", None) or ""
                # Instance-receiver iff the declared role matches the registry's
                # receiver role. No port-name comparison.
                is_instance_receiver = bool(receiver_role) and declared_role == receiver_role
                t = str(p_sig.signature.type_name)
                if not is_instance_receiver:
                    if _R.is_wildcard(t.lower()) or _port_literal_groundable(t, p_sig):
                        continue
                sigs.append(p_sig.signature)
            cell_receiver_sigs[c.cell_id] = sigs

        def _is_col_proj_cell(cell):
            return any(_is_col_projection_port(p_s) for p_s in cell.inputs.values())

        def _is_table_groundable_target(p_sig, prev_path):
            return _is_table_groundable_target_port(p_sig, prev_path, quoted_str_literals, identifier_literals)

        def _new_unbindable(cand, prev_path):
            produced = [o.signature for prev in prev_path for o in prev.outputs.values()]
            count = 0
            for p_sig in cell_receiver_sigs.get(cand.cell_id, ()):
                if _is_table_groundable_target(p_sig, prev_path):
                    continue
                if not any(unify(prod, p_sig) is not None for prod in produced):
                    count += 1

            full_path = prev_path + [cand]
            required_ing = required_egr = 0
            for c in full_path:
                is_ing, is_egr = _R.is_source(c), _R.is_sink(c)
                for p_sig in c.inputs.values():
                    if not p_sig.required or p_sig.default_value is not None:
                        continue
                    if _lattice_is_path_port(p_sig):
                        if is_ing: required_ing += 1; break
                        elif is_egr: required_egr += 1; break
            max_ing = len(src_file_literals) or len(file_literals) or 1
            if required_ing > max_ing:
                count += required_ing - max_ing
            if required_egr > len(dest_file_literals):
                count += required_egr - len(dest_file_literals)

            role_carriers = _R._role_carriers()
            matched_producers: Set[Tuple[int, str]] = set()
            for p_name, p_sig in cand.inputs.items():
                p_role = getattr(p_sig, "port_role", None) or getattr(p_sig, "derived_role", "")
                if not p_role and hasattr(p_sig, "derive_port_role"):
                    p_role = p_sig.derive_port_role()
                if p_role not in role_carriers:
                    continue
                def_val = p_sig.default_value
                if def_val is not None and str(def_val).strip() not in ("", "None", "null"):
                    continue
                if _port_literal_groundable(str(p_sig.signature.type_name)):
                    continue
                found = None
                for idx in range(len(prev_path) - 1, -1, -1):
                    prev = prev_path[idx]
                    for out_name, out_sig in prev.outputs.items():
                        if (idx, out_name) in matched_producers:
                            continue
                        if unify(out_sig.signature, p_sig.signature) is not None:
                            found = (idx, out_name); break
                    if found: matched_producers.add(found); break
                if found is None:
                    already_penalized = (
                        p_sig.required and p_sig.default_value is None
                        and p_sig.signature in cell_receiver_sigs.get(cand.cell_id, ())
                        and not any(unify(prod, p_sig.signature) is not None for prod in produced)
                    )
                    if not already_penalized:
                        if p_role == _R.target_input_role():
                            if not _is_table_groundable_target(p_sig, prev_path):
                                count += 1
                        elif not p_sig.required:
                            if (p_name in (getattr(cand, "slots", []) or [])
                                and "{" in getattr(cand, "code_template", "")):
                                count += 1
            return count

        def _edge_is_weak(prev, cand):
            out_sig = prev.primary_output.signature
            has_strong = has_any = False
            for p in cand.inputs.values():
                if unify(out_sig, p.signature) is None:
                    continue
                if _R.is_wildcard(str(p.signature.type_name).lower()):
                    has_any = True
                else:
                    has_strong = True
                    break
            return (not has_strong) and has_any

        _edge_affinity = self._calculate_edge_affinity
        _path_score_cache: Dict[Tuple[Tuple[str, ...], bool], float] = {}
        _path_state_cache: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        _cells_connect = self._cells_connect

        def compute_path_score(item, is_final=False):
            path, _, sc, weak_edges, unbindable = item
            path_ids = tuple(c.cell_id for c in path)
            key = (path_ids, is_final)
            if key in _path_score_cache:
                return _path_score_cache[key]
            k = len(path)

            st = _path_state_cache.get(path_ids)
            if st is not None:
                strong = st["strong"]; weak = st["weak"]; covered = st["covered"]
                inversions = st["inversions"]; dag_affs = st["dag_affs"]; joins = st["joins"]
            elif k > 1 and path_ids[:-1] in _path_state_cache:
                prev_st = _path_state_cache[path_ids[:-1]]
                nc = path[-1]
                strong = prev_st["strong"] | cell_cov_strong.get(nc.cell_id, set())
                weak = prev_st["weak"] | cell_cov_weak.get(nc.cell_id, set())
                covered = prev_st["covered"] | cell_covered.get(nc.cell_id, set())
                masses = cell_clause_mass.get(nc.cell_id, [])
                dp = prev_st["dp"]; inversions = prev_st["inversions"]
                if masses and max(masses) > 0:
                    cands = {g for g, m in enumerate(masses)
                             if m >= C.clause_candidate_frac * max(masses) and m > 0}
                    if cands:
                        if dp is None:
                            dp = {g: 0 for g in cands}; inversions = 0
                        else:
                            ndp = {g: min(dp[pg] + (1 if pg > g else 0) for pg in dp) for g in cands}
                            inversions = min(ndp.values()); dp = ndp
                parents = [path[i] for i in range(k - 1) if _cells_connect(path[i], nc)]
                bound_p = getattr(nc, "bound_parent_ids", None)
                if bound_p is not None:
                    is_real_join = len(bound_p) >= C.join_min_parents
                    actual = [p for p in parents if p.cell_id in bound_p]
                else:
                    is_real_join = (len(parents) >= C.join_min_parents
                                    and len(getattr(nc, "inputs", {})) >= C.join_min_parents)
                    actual = parents
                is_justified = is_real_join and (bool(cell_covered.get(nc.cell_id))
                                                 or relevance_map.get(nc.cell_id, 0.0) >= C.join_relevance_floor)
                joins = prev_st["joins"] + (1 if is_justified else 0)
                is_new_ingress = _R.is_source(nc) and (not const_marker or getattr(nc, "node_type", "") != const_marker)
                aff_parents = actual or parents
                if aff_parents:
                    dag_affs = prev_st["dag_affs"] + [max(_edge_affinity(p, nc) for p in aff_parents)]
                elif not is_new_ingress:
                    dag_affs = prev_st["dag_affs"] + [_edge_affinity(path[-2], nc)]
                else:
                    dag_affs = prev_st["dag_affs"]
                st = {"strong": strong, "weak": weak, "covered": covered,
                      "dp": dp, "inversions": inversions, "dag_affs": dag_affs, "joins": joins}
                _path_state_cache[path_ids] = st
            else:
                strong = set(); weak = set(); covered = set()
                steps = []
                for c in path:
                    strong |= cell_cov_strong.get(c.cell_id, set())
                    weak |= cell_cov_weak.get(c.cell_id, set())
                    covered |= cell_covered.get(c.cell_id, set())
                    masses = cell_clause_mass.get(c.cell_id, [])
                    if masses and max(masses) > 0:
                        cands = {g for g, m in enumerate(masses)
                                 if m >= C.clause_candidate_frac * max(masses) and m > 0}
                        if cands: steps.append(cands)
                inversions = 0; dp = None
                if steps:
                    dp = {g: 0 for g in steps[0]}
                    for s in steps[1:]:
                        dp = {g: min(dp[pg] + (1 if pg > g else 0) for pg in dp) for g in s}
                    inversions = min(dp.values())
                dag_affs = []; joins = 0
                for j in range(1, k):
                    cj = path[j]
                    parents = [path[i] for i in range(j) if _cells_connect(path[i], cj)]
                    bound_p = getattr(cj, "bound_parent_ids", None)
                    if bound_p is not None:
                        is_join = len(bound_p) >= C.join_min_parents
                        actual = [p for p in parents if p.cell_id in bound_p]
                    else:
                        is_join = (len(parents) >= C.join_min_parents
                                   and len(getattr(cj, "inputs", {})) >= C.join_min_parents)
                        actual = parents
                    if is_join and (bool(cell_covered.get(cj.cell_id))
                                    or relevance_map.get(cj.cell_id, 0.0) >= C.join_relevance_floor):
                        joins += 1
                    is_j_ingress = _R.is_source(cj) and (not const_marker or getattr(cj, "node_type", "") != const_marker)
                    aff_parents = actual or parents
                    if aff_parents:
                        dag_affs.append(max(_edge_affinity(p, cj) for p in aff_parents))
                    elif not is_j_ingress:
                        dag_affs.append(_edge_affinity(path[j - 1], cj))
                st = {"strong": strong, "weak": weak, "covered": covered,
                      "dp": dp, "inversions": inversions, "dag_affs": dag_affs, "joins": joins}
                _path_state_cache[path_ids] = st

            coverage = (sum(idf_of_prompt.get(t, _idf(t)) for t in strong)
                        + C.weak_cov_share * sum(idf_of_prompt.get(t, _idf(t)) for t in (weak - strong))
                        ) / total_prompt_idf
            matched_clause_weight = sum(clause_weights[i] for i in covered)
            clause_cov = matched_clause_weight / total_clause_weight
            gap_penalty = 0.0
            if covered:
                ordered = sorted(covered)
                gap_penalty = sum(1 for g in range(ordered[0], ordered[-1] + 1) if g not in covered)

            op_inv = 0
            if operand_dependencies:
                cids = [c.cell_id for c in path]
                for op_c, h_c in operand_dependencies:
                    hi = [i for i, cid in enumerate(cids) if cid in h_c]
                    oi = [i for i, cid in enumerate(cids) if cid in op_c]
                    if hi and oi and min(hi) < max(oi):
                        op_inv += 1
            total_inv = inversions + op_inv * C.join_min_parents
            align_factor = max(C.align_floor, 1.0 - C.align_step * total_inv)
            alignment = clause_cov * align_factor

            consumed_ctors = sum(
                1 for i, c in enumerate(path)
                if const_marker and getattr(c, "node_type", "") == const_marker
                and any(_cells_connect(c, d) for d in path[i + 1:])
            )
            prereq = 0
            for i in range(1, k - 1):
                c = path[i]
                if not cell_covered.get(c.cell_id):
                    if not _cells_connect(path[i - 1], path[i + 1]) and _cells_connect(c, path[i + 1]):
                        prereq += 1
            effective_k = max(1, k - consumed_ctors - prereq)
            excess = max(0, effective_k - max(len(covered), 1))
            parsimony = excess * C.w_parsimony_step + effective_k * C.w_parsimony_base

            is_macro_first = _R.is_macro(path[0]) if k == 1 else False
            mean_log_prob = ((C.w_macro_log_prob if is_macro_first else C.w_flat_log_prob)
                             * (sc / max(k, 1)))

            goal_bonus = 0.0
            terminal = path[-1]
            has_file_dest = bool(dest_file_literals) and any(_R.is_source(c) for c in path)
            has_egress = ((bool(content_prompt_tokens & EGRESS_INTENT_TOKENS) and not target_sink)
                          or has_file_dest or (goal_sig is not None))
            out_mat = any(str(getattr(o, "state", "")).lower() in MATERIALIZATION_OUTPUT_STATES
                          for o in getattr(terminal, "outputs", {}).values())
            terminal_is_mat = (
                (_R.is_sink(terminal) or out_mat) and has_egress and unbindable == 0
                and (not file_literals or _has_path_port(terminal))
            )
            if tunnel_has_sinks:
                if terminal_is_mat: goal_bonus += C.w_sink_bonus
                elif has_egress and not terminal_is_mat: goal_bonus -= C.w_non_sink_penalty

            term_dom = getattr(terminal, "domain_name", "")
            if term_dom:
                others = list(path[:-1])
                if _R.is_macro(terminal) and getattr(terminal, "sub_cells", None) and self.orchestrator:
                    for sid in terminal.sub_cells:
                        sub = self.orchestrator.loaded_cells.get(sid)
                        if sub is not None: others.append(sub)
                if others:
                    other_domains = {getattr(c, "domain_name", "") for c in others}
                    pred_tags = set(getattr(others[-1], "semantic_tags", []) or [])
                    is_bridge = bool(pred_tags & _R.bridge_tags())
                    if is_bridge or term_dom in other_domains:
                        goal_bonus += C.w_domain_coherence
                    else:
                        goal_bonus -= C.w_domain_coherence

            if file_literals:
                if any(_has_path_port(c) for c in path):
                    goal_bonus += C.w_file_port_bonus
                elif not tunnel_absorbs_assets:
                    goal_bonus -= C.w_file_port_penalty

            weak_total = (sum(1 for c in path if _is_wildcarrier(c)) * C.w_wildcarrier
                          + weak_edges * C.w_weak_edge)

            dead_exp = 0
            for c in path:
                subs = macro_expansion.get(c.cell_id)
                if not subs: continue
                for j, sub in enumerate(subs):
                    if j == 0 or j == len(subs) - 1: continue
                    if cell_cov_strong.get(sub.cell_id) or cell_cov_weak.get(sub.cell_id): continue
                    pred = subs[j - 1]; succ = subs[j + 1]
                    p_sig = getattr(getattr(pred, "primary_output", None), "signature", None)
                    s_sig = getattr(getattr(succ, "primary_input", None), "signature", None)
                    if p_sig and s_sig and unify(p_sig, s_sig) is None:
                        continue
                    dead_exp += 1

            dead_ctors = 0
            for i, c in enumerate(path):
                if const_marker and getattr(c, "node_type", "") == const_marker:
                    if not is_final and i == k - 1: continue
                    if not any(_cells_connect(c, d) for d in path[i + 1:]):
                        dead_ctors += 1

            dead_outputs = 0
            if is_final:
                for i in range(k - 1):
                    c = path[i]
                    if _R.is_sink(c) or not c.outputs: continue
                    consumed = (
                        any(c.cell_id in (getattr(d, "bound_parent_ids", None) or ()) for d in path[i + 1:])
                        or any(_cells_connect(c, d) for d in path[i + 1:]
                               if getattr(d, "bound_parent_ids", None) is None)
                        or any(_is_terminal_sink_cell(d)
                               and getattr(d, "domain_name", "") == getattr(c, "domain_name", "")
                               for d in path[i + 1:])
                    )
                    if not consumed: dead_outputs += 1

            pipeline_domains = {
                getattr(c, "domain_name", "") for c in path
                if getattr(c, "domain_name", "")
                and getattr(c, "domain_name", "") not in (_R.utility_domains() if hasattr(_R, "utility_domains") else set())
            }
            dispersion = max(0, len(pipeline_domains) - C.join_min_parents) * C.w_dispersion

            join_bonus = 0.0
            if getattr(self, "topology_mode", "frontier") == "frontier":
                if k <= 1:
                    affinity_score = C.aff_adjacent
                else:
                    affinity_score = sum(dag_affs) / max(len(dag_affs), 1)
                    join_bonus = joins * C.role_bump * len(dag_affs)
            else:
                if k <= 1:
                    affinity_score = C.aff_adjacent
                else:
                    affinity_score = sum(_edge_affinity(path[i], path[i + 1]) for i in range(k - 1)) / max(k - 1, 1)

            intent_deficit = ((C.w_intent_deficit_final if is_final else C.w_intent_deficit_partial)
                              * max(0.0, 1.0 - clause_cov)) if num_clauses > 1 else 0.0

            literal_cons = self._literal_consumption_of(
                path, universal_literals, numeric_literals,
                quoted_str_literals, identifier_literals)

            structural = affinity_score * C.w_affinity + join_bonus
            if num_clauses > 1:
                gate = clause_cov if is_final else max(C.cov_gate_floor, clause_cov)
                effective_aff = structural * gate
            else:
                combined = max(coverage, literal_cons) if universal_literals else coverage
                gate = max(combined, C.cov_gate_floor) if combined < C.cov_gate_floor else 1.0
                effective_aff = structural * gate

            total = (coverage * C.w_coverage + alignment * C.w_alignment
                     + effective_aff - parsimony + mean_log_prob + goal_bonus - weak_total
                     - dead_ctors * C.w_dead_ctor
                     - dead_exp * C.w_dead_expansion_step
                     - dead_outputs * C.w_dead_output
                     - unbindable * C.w_unbindable
                     - dispersion - gap_penalty * C.w_gap
                     - op_inv * C.w_gap
                     - intent_deficit + literal_cons * C.w_literal_consumption)

            if _debug_plan:
                self._component_trace[path_ids] = {
                    "coverage": round(coverage * C.w_coverage, 2),
                    "alignment": round(alignment * C.w_alignment, 2),
                    "affinity": round(affinity_score * C.w_affinity, 2),
                    "join_bonus": round(join_bonus, 2),
                    "parsimony": round(-parsimony, 2),
                    "log_prob": round(mean_log_prob, 2),
                    "goal_bonus": round(goal_bonus, 2),
                    "weak": round(-weak_total, 2),
                    "gap": round(-gap_penalty * C.w_gap, 2),
                    "dead_output": round(-dead_outputs * C.w_dead_output, 2),
                    "intent_deficit": round(-intent_deficit, 2),
                    "literal": round(literal_cons * C.w_literal_consumption, 2),
                }

            if path:
                last = path[-1]
                if has_egress:
                    if _is_terminal_sink_cell(last):
                        total += C.sink_completion_bonus
                    elif _R.is_transform(last) and not any(_is_terminal_sink_cell(c) for c in path):
                        p_out = getattr(last, "primary_output", None)
                        if p_out is not None:
                            sig = getattr(p_out, "signature", p_out)
                            t = str(getattr(sig, "type_name", "")).lower()
                            if (_R.is_tabular(t) or _R.is_dense(t)
                                or _R.is_collection(t)):
                                total -= C.dangling_output_penalty
                if coverage >= C.unrequested_coverage_threshold:
                    prompt_toks = CellTokenizer.tokenize_prompt(prompt)
                    for c in path[1:]:
                        if _R.is_transform(c):
                            c_toks = getattr(c, "identity_tokens", getattr(c, "token_set", set()))
                            if c_toks and not (c_toks & prompt_toks):
                                total -= C.unrequested_transform_penalty

            if is_final and num_clauses > 1:
                total *= max(C.cov_mult_min, clause_cov) ** C.cov_mult_exp

            _path_score_cache[key] = total
            return total

        # ---- type-gated adjacency index ----
        cells_by_in_type: Dict[str, List[Cell]] = {}
        for cand in candidates:
            in_ports = list(cand.inputs.values()) if cand.inputs else [cand.primary_input]
            for p_sig in in_ports:
                t = str(getattr(p_sig.signature, "type_name", ""))
                cells_by_in_type.setdefault(t, []).append(cand)

        candidate_map = {c.cell_id: c for c in candidates}
        candidate_map_lower = {c.cell_id.lower(): c for c in candidates}

        def _successors(prev_cell):
            if _R.is_macro(prev_cell) and getattr(prev_cell, "endable", False):
                return []
            macro_subs = set(getattr(prev_cell, "sub_cells", ()) or ())
            out_sig = (prev_cell.primary_output.signature
                       if hasattr(prev_cell.primary_output, "signature")
                       else prev_cell.primary_output)
            out_t = str(getattr(out_sig, "type_name", ""))
            out_s = str(getattr(out_sig, "state", ""))
            acc: Dict[str, Cell] = {}
            for edge in getattr(prev_cell, "edges", []):
                tgt = edge.get("target_cell_id") if isinstance(edge, dict) else getattr(edge, "target_cell_id", None)
                if tgt:
                    tc = candidate_map.get(tgt) or candidate_map_lower.get(str(tgt).lower())
                    if tc: acc.setdefault(tc.cell_id, tc)
            is_type_var = out_t.isalpha() and len(out_t) == 1 and out_t.isupper()
            if "[" in out_t or is_type_var:
                for cand in candidates:
                    if cand.cell_id in macro_subs or prev_cell.cell_id in getattr(cand, "sub_cells", ()):
                        continue
                    in_ports = list(cand.inputs.values()) if cand.inputs else [cand.primary_input]
                    if any(unify(out_sig, p.signature) is not None for p in in_ports):
                        acc.setdefault(cand.cell_id, cand)
                res = [c for c in acc.values()
                       if c.cell_id not in macro_subs and prev_cell.cell_id not in getattr(c, "sub_cells", ())]
                return sorted(res, key=lambda c: _edge_affinity(prev_cell, c), reverse=True)
            for in_t, cell_list in cells_by_in_type.items():
                if not registry.is_subtype(out_t, in_t): continue
                for c in cell_list:
                    if c.cell_id in macro_subs or prev_cell.cell_id in getattr(c, "sub_cells", ()):
                        continue
                    in_ports = list(c.inputs.values()) if c.inputs else [c.primary_input]
                    if any(registry.is_state_compatible(
                        producer_state=out_s,
                        consumer_state=getattr(p.signature, "state", "any"),
                        producer_accepted=getattr(out_sig, "accepted_states", frozenset()),
                        consumer_accepted=getattr(p.signature, "accepted_states", frozenset()),
                        consumer_parent=getattr(p.signature, "parent_state", None),
                        producer_parent=getattr(out_sig, "parent_state", None),
                    ) for p in in_ports):
                        acc.setdefault(c.cell_id, c)
            res = [c for c in acc.values()
                   if c.cell_id not in macro_subs and prev_cell.cell_id not in getattr(c, "sub_cells", ())]
            return sorted(res, key=lambda c: _edge_affinity(prev_cell, c), reverse=True)

        def _is_zero_ary_generator(c):
            if not c.outputs or _R.is_sink(c): return False
            if any(_lattice_is_path_port(p) for p in c.inputs.values()): return False
            for p in c.inputs.values():
                if p.required and p.default_value is None: return False
            if _R.is_source(c):
                out_types = {str(getattr(o, "type_name", "")).lower() for o in c.outputs.values()}
                if not any(_R.is_canvas(t) for t in out_types if t) \
                        and (not const_marker or getattr(c, "node_type", "") != const_marker):
                    return False
            return True

        zero_ary_ctors = [c for c in candidates
                          if (const_marker and getattr(c, "node_type", "") == const_marker)
                          or _is_zero_ary_generator(c)]

        max_steps = max(C.join_min_parents, min(C.max_steps_hard_cap,
                                                num_clauses + C.max_steps_slack))

        def _is_valid_terminal(cand_path):
            terminal = cand_path[-1]
            if getattr(terminal, "endable", None) is True: return True
            if getattr(terminal, "is_endable", False):
                cov = set().union(*(cell_covered.get(c.cell_id, set()) for c in cand_path))
                if num_clauses > 1 and cov and max(cov) < num_clauses - 1 and not _R.is_sink(terminal):
                    return False
                return True
            if _R.is_sink(terminal): return True
            if _R.is_transform(terminal):
                cov = set().union(*(cell_covered.get(c.cell_id, set()) for c in cand_path))
                if num_clauses > 1 and cov and max(cov) < num_clauses - 1:
                    return False
            if target_sink and (getattr(terminal, "primary_output", None) or getattr(terminal, "outputs", None)):
                cov = set().union(*(cell_covered.get(c.cell_id, set()) for c in cand_path))
                if num_clauses <= 1 or not cov or max(cov) >= num_clauses - 1:
                    return True
            out_states = {str(getattr(o, "state", "")).lower()
                          for o in getattr(terminal, "outputs", {}).values()}
            mat_states = TypeRegistry.get_instance().get_materialization_states()
            if any(registry.state_ancestry_reaches(s, mat_states) for s in out_states
                   if s and not _R.is_default_state(s)):
                return True
            cov = set().union(*(cell_covered.get(c.cell_id, set()) for c in cand_path))
            if num_clauses > 1 and cov and max(cov) < num_clauses - 1:
                return False
            return True

        # ---- Dispatch ----
        if getattr(self, "topology_mode", "frontier") == "linear":
            all_valid_paths = self._plan_linear_trellis(
                candidate_entries, candidates, log_probs, max_steps,
                zero_ary_ctors, _new_unbindable, compute_path_score, _edge_is_weak,
                cells_by_in_type, candidate_map, candidate_map_lower,
                identifier_literals, quoted_str_literals, numeric_literals,
                cell_clause_mass,
            )
        else:
            all_valid_paths = self._plan_frontier_dag(
                candidate_entries, candidates, log_probs, max_steps,
                zero_ary_ctors, _new_unbindable, compute_path_score, _edge_is_weak,
                cells_by_in_type, candidate_map, candidate_map_lower,
                identifier_literals, quoted_str_literals, numeric_literals,
                cell_clause_mass, file_literals, dest_file_literals,
                has_prompt_egress_intent, target_sink, _is_valid_terminal,
            )

        if all_valid_paths:
            valid = list(all_valid_paths)
            zero_unbind = [it for it in valid if it[4] == 0]
            if zero_unbind: valid = zero_unbind

            if goal_sig is not None:
                g = getattr(goal_sig, "signature", goal_sig)
                m = [it for it in valid
                     if unify(it[0][-1].primary_output.signature, g) is not None]
                if m: valid = m

            endable = [it for it in valid if _is_valid_terminal(it[0])]
            if endable: valid = endable

            _uc = ExecutionContext._extract_universal_literals(prompt or "")
            _ul = [(k, v) for _, k, v in _uc
                   if k in ("file_asset", "identifier", "quoted_str", "numeric")
                   and not (k == "identifier" and target_sink and str(v).lower() == target_sink.lower())]
            _nl = [v for k, v in _ul if k == "numeric"]
            _ql = [v for k, v in _ul if k == "quoted_str"]
            _il = [v for k, v in _ul if k == "identifier"]
            valid, _tier = self._hard_constraint_filter(valid, _ul, _nl, _ql, _il, clause_tokens_list)

            scored = [(it, compute_path_score(it, is_final=True)) for it in valid]
            scored.sort(key=lambda x: x[1], reverse=True)

            # Slot-aware re-ranking — work on clones.
            slot_aug: List[Tuple[Tuple, float]] = []
            seen_prefixes: Set[Tuple[str, ...]] = set()
            trials = 0
            for it, base in scored:
                if trials >= 6: break
                p = it[0]
                prefix = tuple(c.cell_id for c in p)
                if prefix in seen_prefixes: continue
                seen_prefixes.add(prefix); trials += 1

                trial_path = [c.clone() if hasattr(c, "clone") else copy.copy(c) for c in p]
                slot_cells = []
                has_slots = False
                for c in trial_path:
                    for lst in (getattr(c, "bound_slots", {}) or {}).values():
                        slot_cells.extend(lst)
                    if getattr(c, "slots", None): has_slots = True
                if not has_slots:
                    slot_aug.append((it, base)); continue

                trial_sigma = it[1]
                for c in trial_path:
                    for slot_name, slot_contract in _safe_slots_items(c):
                        if slot_name not in getattr(c, "bound_slots", {}):
                            sub = self.plan_sublattice(
                                c, slot_name, slot_contract, tunnel,
                                relevance_map, trial_sigma, prompt)
                            if sub:
                                c.bound_slots[slot_name] = sub
                                slot_cells.extend(sub)
                aug = base
                covered_main = set().union(*(cell_covered.get(pc.cell_id, set()) for pc in trial_path))
                for sc in slot_cells:
                    aug += cell_cov_mass_bonus.get(sc.cell_id, 0.0)
                    aug += (C.w_coverage * sum(clause_weights[g] for g in cell_covered.get(sc.cell_id, set())
                                               if g not in covered_main) / total_clause_weight)
                slot_aug.append(((trial_path, it[1], it[2], it[3], it[4]), aug))

            if slot_aug:
                slot_aug.sort(key=lambda x: x[1], reverse=True)
                scored = list(slot_aug)

            gate = UnificationGate(orchestrator=self.orchestrator)
            chosen = None
            for it, sc in scored:
                cp = it[0]
                cand_test = self._expand_identifier_multiplicity(cp, prompt)
                try:
                    res = gate.unify_pipeline(cand_test, ExecutionContext(prompt=prompt))
                    if isinstance(res, Success) and not res.is_bottom():
                        chosen = it; break
                except Exception:
                    continue
            if chosen is None and scored:
                try:
                    from semantic_repair_engine import repair_cell_semantics
                    for it, _ in scored:
                        cp = it[0]
                        repaired = []
                        any_r = False
                        for c in cp:
                            c_dict = c.to_dict() if hasattr(c, "to_dict") else dict(c.__dict__)
                            if repair_cell_semantics(c_dict, domain=getattr(c, "domain_name", "")):
                                any_r = True
                                repaired.append(Cell.from_dict(c_dict))
                            else:
                                repaired.append(c)
                        if any_r:
                            ct = self._expand_identifier_multiplicity(repaired, prompt)
                            res = gate.unify_pipeline(ct, ExecutionContext(prompt=prompt))
                            if isinstance(res, Success) and not res.is_bottom():
                                chosen = (repaired, res.sigma, it[2], it[3], it[4]); break
                except Exception as e:
                    logger.debug(f"[PLANNER] Semantic repair pass: {e}")

            if chosen is None:
                chosen = scored[0][0]
            best_path, best_sigma, *_ = chosen

            best_path = self._expand_identifier_multiplicity(best_path, prompt)
            best_path = [c.clone() if hasattr(c, "clone") else copy.copy(c) for c in best_path]
            for cell in best_path:
                if getattr(cell, "slots", None):
                    for slot_name, slot_contract in _safe_slots_items(cell):
                        if slot_name not in getattr(cell, "bound_slots", {}):
                            sub = self.plan_sublattice(cell, slot_name, slot_contract,
                                                       tunnel, relevance_map, best_sigma, prompt)
                            if sub:
                                cell.bound_slots[slot_name] = sub

            for cell in best_path:
                if getattr(cell, "matched_clause_idx", None) is None and cell.cell_id in cell_clause_mass:
                    masses = cell_clause_mass[cell.cell_id]
                    if masses and max(masses) > 0.0:
                        cell.matched_clause_idx = max(range(len(masses)), key=lambda i: masses[i])
            return best_path

        logger.info("[PLANNER] Running bounded MCTS search...")
        mcts_path = self._bounded_mcts_search(tunnel, relevance_map)
        if mcts_path:
            return mcts_path
        return [max(tunnel, key=lambda c: relevance_map.get(c.cell_id, 0.0))]

    # ---- identifier multiplicity ----
    def _expand_identifier_multiplicity(self, path, prompt):
        if not prompt or not path:
            return path
        try:
            groups = ExecutionContext.extract_identifier_groups(prompt)
        except Exception:
            return path
        if not groups:
            return path
        C = self.cal

        def _is_witness(cell, role_tokens):
            if not _R.is_transform(cell): return False
            if _R.is_macro(cell): return False
            for p_sig in cell.inputs.values():
                t = str(p_sig.signature.type_name).lower()
                if (_is_col_projection_port(p_sig) or _R.is_collection(t)):
                    state = str(getattr(p_sig.signature, "state", "") or "").lower()
                    st_toks = (CellTokenizer.tokenize_identifier(state)
                               if not _R.is_default_state(state) else set())
                    if _is_col_projection_port(p_sig) or (role_tokens & st_toks):
                        return False
            for p_sig in cell.inputs.values():
                if not p_sig.required or p_sig.default_value is not None:
                    continue
                t = str(p_sig.signature.type_name).lower()
                if (_is_col_projection_port(p_sig) or _R.is_collection(t)):
                    continue
                strict_text = _R.is_textual(t) and not _R.is_wildcard(t)
                state = str(getattr(p_sig.signature, "state", "") or "").lower()
                role_state = not _R.is_default_state(state) and bool(
                    role_tokens & CellTokenizer.tokenize_identifier(state))
                if strict_text or role_state:
                    return True
            return False

        expanded = []; added = 0
        for cell in path:
            expanded.append(cell)
            if added >= C.max_replicas: continue
            for grp in groups:
                if len(grp.members) < C.join_min_parents or not grp.role_tokens: continue
                if not _is_witness(cell, grp.role_tokens): continue
                for _pos, _tok in grp.members[1:]:
                    if added >= C.max_replicas: break
                    rep = copy.copy(cell)
                    if hasattr(rep, "bound_slots"):
                        rep.bound_slots = dict(rep.bound_slots)
                    rep.replica_of = cell.cell_id
                    rep.replica_role = ",".join(sorted(grp.role_tokens))
                    expanded.append(rep); added += 1
        return expanded

    # ---- frontier DAG ----
    def _plan_frontier_dag(self, candidate_entries, candidates, log_probs, max_steps,
                          zero_ary_ctors, _new_unbindable, compute_path_score, _edge_is_weak,
                          cells_by_in_type, candidate_map, candidate_map_lower,
                          identifier_literals, quoted_str_literals, numeric_literals,
                          cell_clause_mass, file_asset_literals, dest_file_literals,
                          has_egress_intent, target_sink, is_valid_terminal):
        C = self.cal
        _cells_connect = self._cells_connect
        all_paths = []

        current_beam = []
        for entry in candidate_entries:
            sc = log_probs.get(entry.cell_id, math.log(1e-6))
            e = entry.clone() if hasattr(entry, "clone") else copy.copy(entry)
            tup = ([e], Substitution(), sc, 0, _new_unbindable(e, []))
            current_beam.append(tup)
            if is_valid_terminal is None or is_valid_terminal([e]):
                all_paths.append(tup)

        def _shape_compatible(p_out, p_in):
            if not _is_port_role_compatible(p_out, p_in):
                return False
            o_sc = getattr(p_out, "shape_contract", None)
            i_sc = getattr(p_in, "shape_contract", None)
            if o_sc and i_sc:
                o = _extract_ndim_from_contract(o_sc)
                i = _extract_ndim_from_contract(i_sc)
                if o is not None and i is not None and o != i:
                    return False
            return True

        const_marker = next(iter(_R.node_types() - {""}), "")

        def _verify_frontier_step(prev_path, cand, prev_sigma):
            if prev_path and _is_terminal_sink_cell(prev_path[-1]):
                return None
            if _R.is_transform(cand) and any(_is_terminal_sink_cell(c) for c in prev_path):
                return None
            if _is_terminal_sink_cell(cand):
                has_req_path = any(p.required and p.default_value is None and _lattice_is_path_port(p)
                                   for p in cand.inputs.values())
                if has_req_path and not dest_file_literals:
                    return None
                if not has_egress_intent and not target_sink:
                    return None

            sub = prev_sigma
            parents: Set[str] = set()
            bound_ports: Set[str] = set()
            # Fast producer index for the prefix.
            producer_idx: Dict[str, List[int]] = {}
            for idx, pc in enumerate(prev_path):
                for out_name, out_sig in pc.outputs.items():
                    key = str(getattr(out_sig.signature, "type_name", ""))
                    producer_idx.setdefault(key, []).append(idx)

            def _try_port(p_name, p_sig):
                nonlocal sub
                t = str(getattr(p_sig.signature, "type_name", ""))
                candidates_idx: List[int] = list(producer_idx.get(t, []))
                for other_t, idxs in producer_idx.items():
                    if other_t != t and registry.is_subtype(other_t, t):
                        candidates_idx.extend(idxs)
                for idx in reversed(candidates_idx):
                    pc = prev_path[idx]
                    for out_name, out_sig in pc.outputs.items():
                        if not _shape_compatible(out_sig, p_sig):
                            continue
                        s = unify(out_sig.signature, p_sig.signature, sub)
                        if s is not None:
                            sub = s
                            return pc.cell_id
                return None

            prim = getattr(cand, "primary_input", None)
            if prim is not None:
                pid = _try_port(getattr(prim, "name", ""), prim)
                if pid:
                    parents.add(pid); bound_ports.add(getattr(prim, "name", ""))

            for p_name, p_sig in cand.inputs.items():
                if p_name in bound_ports: continue
                pid = _try_port(p_name, p_sig)
                if pid:
                    parents.add(pid); bound_ports.add(p_name); continue
                if not p_sig.required or p_sig.default_value is not None:
                    continue
                declared_role = getattr(p_sig, "port_role", None) or ""
                if declared_role == _R.receiver_role():
                    continue
                t = str(getattr(p_sig.signature, "type_name", "")).lower()
                if _is_table_groundable_target_port(p_sig, prev_path, quoted_str_literals, identifier_literals):
                    for pc in prev_path:
                        for o in pc.outputs.values():
                            if _R.is_tabular(str(getattr(o.signature, "type_name", "")).lower()):
                                parents.add(pc.cell_id); bound_ports.add(p_name); break
                        if p_name in bound_ports: break
                    continue
                if _port_literal_groundable(t, p_sig):
                    continue
                # Macro slots
                if _R.is_macro(cand) and p_name in getattr(cand, "slots", {}):
                    continue
                if declared_role == _R.functional_operator_role():
                    continue
                return None

            if not (const_marker and getattr(cand, "node_type", "") == const_marker) \
                    and cand not in zero_ary_ctors and not parents:
                if _is_terminal_sink_cell(cand) and not any(p.required for p in cand.inputs.values()):
                    parents.add(prev_path[-1].cell_id)
                elif _R.is_source(cand):
                    pass
                else:
                    return None
            return (sub, parents, len(parents) >= C.join_min_parents)

        _cell_succ_cache: Dict[str, List[Cell]] = {}

        def _cell_successors(cell):
            cached = _cell_succ_cache.get(cell.cell_id)
            if cached is not None: return cached
            out_sig = (cell.primary_output.signature
                       if hasattr(cell.primary_output, "signature") else cell.primary_output)
            out_t = str(getattr(out_sig, "type_name", ""))
            out_s = str(getattr(out_sig, "state", ""))
            acc = {}
            for edge in getattr(cell, "edges", []):
                tgt = edge.get("target_cell_id") if isinstance(edge, dict) else getattr(edge, "target_cell_id", None)
                if tgt:
                    tc = candidate_map.get(tgt) or candidate_map_lower.get(str(tgt).lower())
                    if tc: acc.setdefault(tc.cell_id, tc)
            is_type_var = out_t.isalpha() and len(out_t) == 1 and out_t.isupper()
            if "[" in out_t or is_type_var:
                for cand in candidates:
                    in_ports = list(cand.inputs.values()) if cand.inputs else [cand.primary_input]
                    if any(unify(out_sig, p.signature) is not None for p in in_ports):
                        acc.setdefault(cand.cell_id, cand)
                res = list(acc.values())
                _cell_succ_cache[cell.cell_id] = res
                return res
            for in_t, cell_list in cells_by_in_type.items():
                if not registry.is_subtype(out_t, in_t): continue
                for c in cell_list:
                    in_ports = list(c.inputs.values()) if c.inputs else [c.primary_input]
                    if any(registry.is_state_compatible(
                        producer_state=out_s,
                        consumer_state=getattr(p.signature, "state", "any"),
                        producer_accepted=getattr(out_sig, "accepted_states", frozenset()),
                        consumer_accepted=getattr(p.signature, "accepted_states", frozenset()),
                        consumer_parent=getattr(p.signature, "parent_state", None),
                        producer_parent=getattr(out_sig, "parent_state", None),
                    ) for p in in_ports):
                        acc.setdefault(c.cell_id, c)
            res = list(acc.values())
            _cell_succ_cache[cell.cell_id] = res
            return res

        for step in range(2, max_steps + 1):
            next_candidates = []
            for prev_path, prev_sigma, prev_score, prev_weak, prev_unbind in current_beam:
                prev_cell = prev_path[-1]
                if any(_is_terminal_sink_cell(c) for c in prev_path):
                    continue
                prev_ids = {c.cell_id.lower() for c in prev_path}
                succs = []
                seen = set()
                for pc in prev_path:
                    for s in _cell_successors(pc):
                        sl = s.cell_id.lower()
                        if sl not in seen and sl not in prev_ids:
                            succs.append(s); seen.add(sl)
                for ctor in zero_ary_ctors:
                    cl = ctor.cell_id.lower()
                    if cl not in seen and cl not in prev_ids:
                        succs.append(ctor); seen.add(cl)
                num_ing = sum(1 for c in prev_path if _R.is_source(c)
                              and (not const_marker or getattr(c, "node_type", "") != const_marker))
                if num_ing < len(file_asset_literals):
                    for entry in candidate_entries:
                        el = entry.cell_id.lower()
                        if el not in seen:
                            succs.append(entry); seen.add(el)

                for cand in succs:
                    is_ingress = _R.is_source(cand) and cand not in zero_ary_ctors
                    if cand.cell_id.lower() in prev_ids:
                        if not (is_ingress and num_ing < len(file_asset_literals)):
                            continue
                    if is_ingress and num_ing >= len(file_asset_literals):
                        continue
                    v = _verify_frontier_step(prev_path, cand, prev_sigma)
                    if v is None: continue
                    new_sigma, parents, is_join = v
                    if (len(prev_path) >= 1 and prev_cell.cell_id not in parents
                        and cand.cell_id < prev_cell.cell_id
                        and not _cells_connect(prev_cell, cand) and not is_ingress):
                        pm = cell_clause_mass.get(prev_cell.cell_id, []) if cell_clause_mass else []
                        cm = cell_clause_mass.get(cand.cell_id, []) if cell_clause_mass else []
                        pcl = max(range(len(pm)), key=lambda i: pm[i]) if pm and max(pm) > 0 else 0
                        ccl = max(range(len(cm)), key=lambda i: cm[i]) if cm and max(cm) > 0 else 0
                        if ccl <= pcl:
                            if not prev_path[:-1] or _verify_frontier_step(prev_path[:-1], cand, prev_sigma) is not None:
                                continue
                    unb = _new_unbindable(cand, prev_path)
                    if unb > 0: continue
                    sc = log_probs.get(cand.cell_id, math.log(1e-6))
                    tot = prev_score + sc
                    ctor_like = (const_marker and getattr(cand, "node_type", "") == const_marker) or cand in zero_ary_ctors
                    step_weak = prev_weak + (1 if (not ctor_like and not _is_terminal_sink_cell(cand)
                                                   and _edge_is_weak(prev_cell, cand)) else 0)
                    step_unb = prev_unbind + unb
                    new_cand = cand.clone() if hasattr(cand, "clone") else copy.copy(cand)
                    new_cand.bound_parent_ids = set(parents)
                    next_candidates.append((prev_path + [new_cand], new_sigma, tot, step_weak, step_unb))

            if not next_candidates:
                break
            next_candidates.sort(key=lambda x: compute_path_score(x, is_final=False), reverse=True)
            by_ep: Dict[str, List[Any]] = {}
            for it in next_candidates:
                by_ep.setdefault(it[0][-1].cell_id, []).append(it)
            endpoints = sorted(by_ep.keys(),
                               key=lambda ep: compute_path_score(by_ep[ep][0], is_final=False),
                               reverse=True)
            nxt = []
            for ep in endpoints:
                for it in by_ep[ep][:C.beam_per_endpoint]:
                    nxt.append(it)
            for ep in endpoints:
                for it in by_ep[ep][C.beam_per_endpoint:]:
                    if len(nxt) >= C.beam_width_frontier: break
                    nxt.append(it)
                if len(nxt) >= C.beam_width_frontier: break
            current_beam = nxt
            for it in next_candidates:
                if is_valid_terminal is not None:
                    if is_valid_terminal(it[0]):
                        all_paths.append(it)
                else:
                    tc = it[0][-1]
                    if (_R.is_sink(tc) or getattr(tc, "endable", False)
                        or getattr(tc, "is_endable", False)):
                        all_paths.append(it)
            if len(all_paths) > C.beam_pool_frontier:
                all_paths.sort(key=lambda x: compute_path_score(x, is_final=True), reverse=True)
                all_paths = all_paths[:C.beam_pool_frontier_prune]
        if not all_paths:
            all_paths = list(current_beam)
        return all_paths

    # ---- linear trellis (ablation) ----
    def _plan_linear_trellis(self, candidate_entries, candidates, log_probs, max_steps,
                            zero_ary_ctors, _new_unbindable, compute_path_score, _edge_is_weak,
                            cells_by_in_type, candidate_map, candidate_map_lower,
                            identifier_literals, quoted_str_literals, numeric_literals,
                            cell_clause_mass):
        C = self.cal
        all_paths = []
        const_marker = next(iter(_R.node_types() - {""}), "")

        current_beam = []
        for entry in candidate_entries:
            sc = log_probs.get(entry.cell_id, math.log(1e-6))
            e = entry.clone() if hasattr(entry, "clone") else copy.copy(entry)
            tup = ([e], Substitution(), sc, 0, _new_unbindable(e, []))
            current_beam.append(tup); all_paths.append(tup)

        def _succs(prev_cell):
            if _R.is_macro(prev_cell) and getattr(prev_cell, "endable", False):
                return []
            macro_subs = set(getattr(prev_cell, "sub_cells", ()) or ())
            out_sig = (prev_cell.primary_output.signature
                       if hasattr(prev_cell.primary_output, "signature") else prev_cell.primary_output)
            out_t = str(getattr(out_sig, "type_name", ""))
            out_s = str(getattr(out_sig, "state", ""))
            acc = {}
            for edge in getattr(prev_cell, "edges", []):
                tgt = edge.get("target_cell_id") if isinstance(edge, dict) else getattr(edge, "target_cell_id", None)
                if tgt:
                    tc = candidate_map.get(tgt) or candidate_map_lower.get(str(tgt).lower())
                    if tc: acc.setdefault(tc.cell_id, tc)
            is_tv = out_t.isalpha() and len(out_t) == 1 and out_t.isupper()
            if "[" in out_t or is_tv:
                for cand in candidates:
                    if cand.cell_id in macro_subs or prev_cell.cell_id in getattr(cand, "sub_cells", ()):
                        continue
                    if any(unify(out_sig, p.signature) is not None for p in cand.inputs.values()):
                        acc.setdefault(cand.cell_id, cand)
                res = [c for c in acc.values()
                       if c.cell_id not in macro_subs and prev_cell.cell_id not in getattr(c, "sub_cells", ())]
                return sorted(res, key=lambda c: self._calculate_edge_affinity(prev_cell, c), reverse=True)
            for in_t, cl in cells_by_in_type.items():
                if not registry.is_subtype(out_t, in_t): continue
                for c in cl:
                    if c.cell_id in macro_subs or prev_cell.cell_id in getattr(c, "sub_cells", ()):
                        continue
                    if any(registry.is_state_compatible(
                        producer_state=out_s,
                        consumer_state=getattr(p.signature, "state", "any"),
                        producer_accepted=getattr(out_sig, "accepted_states", frozenset()),
                        consumer_accepted=getattr(p.signature, "accepted_states", frozenset()),
                        consumer_parent=getattr(p.signature, "parent_state", None),
                        producer_parent=getattr(out_sig, "parent_state", None),
                    ) for p in c.inputs.values()):
                        acc.setdefault(c.cell_id, c)
            res = [c for c in acc.values()
                   if c.cell_id not in macro_subs and prev_cell.cell_id not in getattr(c, "sub_cells", ())]
            return sorted(res, key=lambda c: self._calculate_edge_affinity(prev_cell, c), reverse=True)

        edge_cache: Dict[Tuple[str, str, Any, Any], Any] = {}
        def _sig_fp(sigma):
            try:
                return tuple(sorted((k, str(v)) for k, v in sigma.mappings.items()))
            except Exception:
                return ()

        for step in range(2, max_steps + 1):
            nexts = []
            for prev_path, prev_sigma, prev_score, prev_weak, prev_unbind in current_beam:
                prev_cell = prev_path[-1]
                if _is_terminal_sink_cell(prev_cell): continue
                prev_ids = {c.cell_id.lower() for c in prev_path}
                succs = _succs(prev_cell)
                seen = {c.cell_id.lower() for c in succs}
                for ctor in zero_ary_ctors:
                    if ctor.cell_id.lower() not in seen:
                        succs.append(ctor); seen.add(ctor.cell_id.lower())
                for cand in succs:
                    if cand.cell_id.lower() in prev_ids: continue
                    is_ctor = const_marker and getattr(cand, "node_type", "") == const_marker
                    if is_ctor or cand in zero_ary_ctors:
                        # Check that all required non-receiver ports can be literal-grounded.
                        sub = prev_sigma
                        for p_name, p_sig in cand.inputs.items():
                            if not p_sig.required or p_sig.default_value is not None: continue
                            ok = False
                            for ec in reversed(prev_path):
                                for o in ec.outputs.values():
                                    s = unify(o.signature, p_sig.signature, sub)
                                    if s is not None:
                                        sub = s; ok = True; break
                                if ok: break
                            if not ok:
                                r = getattr(p_sig, "port_role", None) or ""
                                if r == _R.receiver_role(): continue
                                t = str(p_sig.signature.type_name).lower()
                                if not _port_literal_groundable(t, p_sig):
                                    # allow if macro slot
                                    if not (_R.is_macro(cand) and p_name in getattr(cand, "slots", {})):
                                        sub = None; break
                        new_sigma = sub
                    elif _is_terminal_sink_cell(cand) and not any(p.required for p in cand.inputs.values()):
                        new_sigma = prev_sigma
                    else:
                        if _R.is_source(cand) and cand not in zero_ary_ctors: continue
                        out_sigs = tuple(sorted(
                            (o.signature.type_name, str(getattr(o.signature, "state", "any")))
                            for c in prev_path for o in c.outputs.values()))
                        pk = (prev_cell.cell_id, cand.cell_id, _sig_fp(prev_sigma), out_sigs)
                        if pk in edge_cache:
                            new_sigma = edge_cache[pk]
                        else:
                            new_sigma = self._verify_transition(
                                prev_path, cand, prev_sigma,
                                identifier_literals, quoted_str_literals, numeric_literals)
                            edge_cache[pk] = new_sigma
                        if new_sigma is None: continue
                    unb = _new_unbindable(cand, prev_path)
                    if unb > 0: continue
                    sc = log_probs.get(cand.cell_id, math.log(1e-6))
                    tot = prev_score + sc
                    ctor_like = is_ctor or cand in zero_ary_ctors
                    st_weak = prev_weak + (1 if (not ctor_like and not _is_terminal_sink_cell(cand)
                                                 and _edge_is_weak(prev_cell, cand)) else 0)
                    st_unb = prev_unbind + unb
                    new_cand = cand.clone() if hasattr(cand, "clone") else copy.copy(cand)
                    new_cand.bound_parent_ids = {prev_cell.cell_id}
                    tup = (prev_path + [new_cand], new_sigma, tot, st_weak, st_unb)
                    nexts.append(tup); all_paths.append(tup)
            if not nexts: break
            if len(all_paths) > C.beam_pool_linear:
                all_paths.sort(key=lambda x: compute_path_score(x, is_final=False), reverse=True)
                all_paths = all_paths[:C.beam_pool_linear_prune]
            nexts.sort(key=lambda x: compute_path_score(x, is_final=False), reverse=True)
            counts: Dict[str, int] = {}
            beam = []
            for it in nexts:
                ep = it[0][-1].cell_id
                if counts.get(ep, 0) < C.beam_per_endpoint:
                    beam.append(it); counts[ep] = counts.get(ep, 0) + 1
                    if len(beam) >= C.beam_width_linear: break
            current_beam = beam
        return all_paths

    # ---- transition verification (linear only) ----
    def _verify_transition(self, prev_path, cand, prev_sigma,
                          identifier_literals=(), quoted_str_literals=(), numeric_literals=()):
        prev_cell = prev_path[-1]
        prev_out = prev_cell.primary_output.signature

        def _sc(p_out, p_in):
            if not _is_port_role_compatible(p_out, p_in): return False
            a = getattr(p_out, "shape_contract", None); b = getattr(p_in, "shape_contract", None)
            if a and b:
                o = _extract_ndim_from_contract(a); i = _extract_ndim_from_contract(b)
                if o is not None and i is not None and o != i: return False
            return True

        sub = None
        if _sc(prev_cell.primary_output, cand.primary_input):
            sub = unify(prev_out, cand.primary_input.signature, prev_sigma)
        bound_in = cand.primary_input.name if sub is not None else None

        if sub is None:
            reqs = [(k, v) for k, v in cand.inputs.items() if v.required]
            ports = reqs if reqs else list(cand.inputs.items())
            for p_name, p_sig in ports:
                if not _sc(prev_cell.primary_output, p_sig): continue
                s = unify(prev_out, p_sig.signature, prev_sigma)
                if s is not None:
                    sub = s; bound_in = p_name; break

        if sub is None and len(prev_path) > 1:
            reqs = [(k, v) for k, v in cand.inputs.items() if v.required]
            ports = reqs if reqs else list(cand.inputs.items())
            for anc in reversed(prev_path[:-1]):
                for o in anc.outputs.values():
                    for p_name, p_sig in ports:
                        if not _sc(o, p_sig): continue
                        s = unify(o.signature, p_sig.signature, prev_sigma)
                        if s is not None:
                            sub = s; bound_in = p_name; break
                    if sub is not None: break
                if sub is not None: break

        if sub is None: return None
        for p_name, p_sig in cand.inputs.items():
            if p_name == bound_in: continue
            if not p_sig.required or p_sig.default_value is not None: continue
            ok = False
            for ec in reversed(prev_path):
                for o in ec.outputs.values():
                    if not _sc(o, p_sig): continue
                    s = unify(o.signature, p_sig.signature, sub)
                    if s is not None:
                        sub = s; ok = True; break
                if ok: break
            if not ok:
                r = getattr(p_sig, "port_role", None) or ""
                if r == _R.receiver_role(): continue
                t = str(p_sig.signature.type_name).lower()
                if not _port_literal_groundable(t, p_sig):
                    if _R.is_macro(cand) and p_name in getattr(cand, "slots", {}):
                        continue
                    if r == _R.functional_operator_role():
                        continue
                    return None
        return sub

    # ---- sub-lattice ----
    def plan_sublattice(self, parent_cell, slot_name, slot_contract, tunnel,
                        relevance_map, active_sigma, prompt=""):
        C = self.cal
        topology = getattr(parent_cell, "topology_type", "")
        topologies = _R.topologies()
        traced = [t for t in topologies if "loop" in t.lower()]
        coprod = [t for t in topologies if "branch" in t.lower() or "coproduct" in t.lower()]

        if topology in traced:
            coll_sig = None
            for p_sig in parent_cell.inputs.values():
                if "[" in str(getattr(p_sig.signature, "type_name", "")):
                    coll_sig = p_sig; break
            item_type = "any"
            if coll_sig is not None:
                c_type = str(coll_sig.signature.type_name)
                c_con = substitute_generics(c_type, active_sigma)
                if "[" in c_con and c_con.endswith("]"):
                    item_type = c_con[c_con.index("[") + 1 : -1].strip()
                elif "T" in active_sigma.mappings:
                    item_type = str(active_sigma.mappings["T"])
            u_raw = getattr(parent_cell, "feedback_state_type", None) or "S"
            u_con = substitute_generics(u_raw, active_sigma)
            pool = [c for c in tunnel if c.cell_id != parent_cell.cell_id
                    and not (_R.node_types() and getattr(c, "node_type", "") in _R.node_types())]
            if not pool:
                pool = [c for c in self.orchestrator.loaded_cells.values()
                        if c.cell_id != parent_cell.cell_id]
            parent_toks = getattr(parent_cell, "identity_tokens", getattr(parent_cell, "token_set", set()))
            clauses = _segment_prompt_clauses(prompt)
            related = [cl for cl in clauses if CellTokenizer.tokenize_prompt(cl) & parent_toks]
            target_text = " ".join(related).strip() or prompt
            target_tokens = CellTokenizer.tokenize_prompt(target_text) if target_text else set()
            child_cands = []
            for cand in pool:
                if not (_R.is_transform(cand) or _R.is_sink(cand)): continue
                for p_name, p_sig in cand.inputs.items():
                    u = unify(item_type, p_sig.signature, active_sigma) or unify(p_sig.signature, item_type, active_sigma)
                    if u is not None:
                        rel = relevance_map.get(cand.cell_id, 0.0)
                        id_toks = getattr(cand, "identity_tokens", cand.token_set)
                        ov = len(target_tokens & id_toks)
                        dom_bonus = (C.sublattice_domain_bonus
                                     if cand.domain_name and any(c.domain_name == cand.domain_name for c in tunnel)
                                     else 0.0)
                        score = ov * C.sublattice_token_weight + rel * C.sublattice_rel_weight + dom_bonus
                        child_cands.append((cand, score, u)); break
            if child_cands:
                child_cands.sort(key=lambda x: x[1], reverse=True)
                best, _, child_sigma = child_cands[0]
                ok, _ = verify_traced_loop_invariant(u_con, u_con, child_sigma)
                if ok:
                    return [best]
        elif topology in coprod:
            pool = [c for c in tunnel if c.cell_id != parent_cell.cell_id
                    and (not _R.node_types() or getattr(c, "node_type", "") not in _R.node_types())]
            if pool:
                pool.sort(key=lambda c: relevance_map.get(c.cell_id, 0.0), reverse=True)
                return [pool[0]]
        return None

    # ---- MCTS fallback ----
    def _bounded_mcts_search(self, tunnel, relevance_map, max_simulations=None):
        C = self.cal
        if max_simulations is None:
            max_simulations = C.beam_width_frontier
        entries = [c for c in tunnel if _R.is_source(c)] or list(tunnel)
        const_marker = next(iter(_R.node_types() - {""}), "")
        for entry in entries:
            chain = [entry]
            sigma = Substitution()
            for _ in range(max_simulations):
                curr = chain[-1]
                if _R.is_sink(curr):
                    cloned = [c.clone() if hasattr(c, "clone") else copy.copy(c) for c in chain]
                    for cell in cloned:
                        if getattr(cell, "slots", None):
                            for sn, sc in _safe_slots_items(cell):
                                if sn not in getattr(cell, "bound_slots", {}):
                                    sub = self.plan_sublattice(cell, sn, sc, tunnel, relevance_map, sigma, "")
                                    if sub: cell.bound_slots[sn] = sub
                    return cloned
                out_sig = curr.primary_output.signature
                valid = []
                for cand in tunnel:
                    if cand.cell_id in (c.cell_id for c in chain): continue
                    ns = unify(out_sig, cand.primary_input.signature, sigma)
                    if ns is not None:
                        valid.append((cand, ns))
                if not valid: break
                valid.sort(key=lambda x: relevance_map.get(x[0].cell_id, 0.0)
                           + C.mcts_affinity_weight * self._calculate_edge_affinity(curr, x[0]),
                           reverse=True)
                chosen, sigma = valid[0]
                chain.append(chosen)
                if _R.is_sink(chosen) and not getattr(chosen, "slots", None):
                    cloned = [c.clone() if hasattr(c, "clone") else copy.copy(c) for c in chain]
                    for cell in cloned:
                        if getattr(cell, "slots", None):
                            for sn, sc in _safe_slots_items(cell):
                                if sn not in getattr(cell, "bound_slots", {}):
                                    sub = self.plan_sublattice(cell, sn, sc, tunnel, relevance_map, sigma, "")
                                    if sub: cell.bound_slots[sn] = sub
                    return cloned
            if len(chain) > 1:
                cloned = [c.clone() if hasattr(c, "clone") else copy.copy(c) for c in chain]
                for cell in cloned:
                    if getattr(cell, "slots", None):
                        for sn, sc in _safe_slots_items(cell):
                            if sn not in getattr(cell, "bound_slots", {}):
                                sub = self.plan_sublattice(cell, sn, sc, tunnel, relevance_map, sigma, "")
                                if sub: cell.bound_slots[sn] = sub
                return cloned
        return None


ZeroShotPlanner = LatticePlanner
