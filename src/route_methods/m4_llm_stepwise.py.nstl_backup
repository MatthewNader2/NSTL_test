"""
src/route_methods/m4_llm_stepwise.py - Neuro-Symbolic Topological Lattice (NSTL)
M4: Step-wise LLM Guided Next-Cell Proposal with Monadic Lattice Validation.
"""

from __future__ import annotations
import json
import re
from typing import Dict, List, Optional, Set, Tuple, Any

from .base import RouteMethod, STOPWORDS

try:
    from ..lattice import Cell, LatticeOrchestrator, TypeRegistry
    from ..tokenizer import CellTokenizer
    from ..unification import unify, ExecutionContext
    from ..inference import ModelManager
    from ..planner import _is_terminal_sink_cell
except (ImportError, ValueError):
    from lattice import Cell, LatticeOrchestrator, TypeRegistry
    from tokenizer import CellTokenizer
    from unification import unify, ExecutionContext
    from inference import ModelManager
    from planner import _is_terminal_sink_cell


class M4LLMStepwiseRouteMethod(RouteMethod):
    """
    RouteMethod M4: Step-wise LLM Guidance.
    At each pipeline progression point, queries the LLM to propose the next cell ID
    from a set of structurally valid candidate continuations, with strict monadic validation.
    """
    name: str = "m4_llm_stepwise"

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
        orch = orchestrator or self.orchestrator
        if not tunnel:
            return []
        if len(tunnel) == 1:
            return [tunnel[0]]

        candidates = [c for c in tunnel if getattr(c, "node_type", "") != "constant"]
        if not candidates:
            return [tunnel[0]]

        cand_map = {c.cell_id: c for c in candidates}
        mm = ModelManager.get_instance()
        has_llm = mm.active_profile is not None and mm.can_synthesize()

        # 1. Entry Selection
        prompt_tokens = CellTokenizer.tokenize_prompt(prompt)
        l0_extracted = ExecutionContext._extract_universal_literals(prompt or "") if ctx or prompt else []
        file_literals = [v for _, t, v in l0_extracted if t == "file_asset"]

        stage1_cands = [c for c in candidates if getattr(c, "stage", None) == 1]
        entry_pool = stage1_cands if stage1_cands else candidates[:5]

        def _score_entry(c: Cell) -> float:
            sc = relevance_map.get(c.cell_id, 0.0) * 5.0 + len(prompt_tokens & getattr(c, "identity_tokens", c.token_set)) * 3.0
            is_path_consumer = any(
                getattr(p, "abstract_type", None) == "path"
                or getattr(p, "port_role", None) in ("source_data", "model_sink")
                or getattr(p, "derived_role", None) in ("source_data", "model_sink")
                or getattr(p.signature, "abstract_type", None) == "path"
                for p in c.inputs.values()
            )
            if file_literals and is_path_consumer:
                sc += 5.0
            return sc

        best_entry = max(entry_pool, key=_score_entry)

        path: List[Cell] = [best_entry]
        visited_ids: Set[str] = {best_entry.cell_id}

        clauses = self.segment_prompt_clauses(prompt)
        num_clauses = len(clauses) if clauses else 1
        dynamic_cap = max(max_transforms + 2, num_clauses + 3)
        max_steps = max(2, min(16, dynamic_cap))

        registry = TypeRegistry.get_instance()

        # Declared Task Polarity: detect regression vs classification intent
        reg_tokens = {"regression", "regressor", "continuous"}
        cls_tokens = {"classification", "classifier", "categorical", "discrete"}
        has_reg_intent = bool(prompt_tokens & reg_tokens) and not bool(prompt_tokens & cls_tokens)
        has_cls_intent = bool(prompt_tokens & cls_tokens) and not bool(prompt_tokens & reg_tokens)

        def _covered_clauses_count(p_cells: List[Cell]) -> int:
            covered = 0
            for cl in clauses:
                cl_toks = CellTokenizer.tokenize_prompt(cl)
                if any(len(cl_toks & getattr(c, "identity_tokens", c.token_set)) > 0 for c in p_cells):
                    covered += 1
            return covered

        prev_cov = _covered_clauses_count(path)
        stall_count = 0

        # 2. Step-wise continuation loop
        for step in range(max_steps):
            curr = path[-1]
            if getattr(curr, "stage", None) == 3 and len(path) > 1:
                break

            # Sinks without sublattice slots conclude the path (D5)
            if any(_is_terminal_sink_cell(c) for c in path):
                break

            curr_out_st = str(getattr(curr.primary_output, "state", "")).lower()
            curr_props = registry.get_state_properties(curr_out_st)
            is_curr_prediction = bool(curr_props.get("is_prediction")) or curr_out_st in (
                "predicted_values", "predicted_labels", "predicted_probabilities"
            )

            valid_next = []
            for c in candidates:
                if c.cell_id in visited_ids:
                    continue
                if not self.step_unifies(curr, c, prev_path=path):
                    continue

                cand_out_st = str(getattr(c.primary_output, "state", "")).lower()
                # 1. Declared Task Polarity Check
                if has_reg_intent:
                    if cand_out_st in ("predicted_labels", "cluster_labels") or "label" in cand_out_st:
                        continue
                elif has_cls_intent:
                    if cand_out_st == "predicted_values":
                        continue

                # 2. Forbid Fitting on Predictions: Estimator cannot consume prediction outputs
                is_cand_estimator = getattr(c, "node_role", "") in ("estimator", "model") or "fit" in c.cell_id.lower()
                if is_curr_prediction and is_cand_estimator:
                    continue

                valid_next.append(c)

            if not valid_next:
                break

            def _score_next(cand: Cell) -> float:
                rel = relevance_map.get(cand.cell_id, 0.0)
                aff = self.calculate_edge_affinity(curr, cand, orch, relevance_map=relevance_map)
                tok_bonus = len(cand.token_set & prompt_tokens) * 3.0
                cand_stage = getattr(cand, "stage", 2) or 2
                is_path_consumer = any(
                    getattr(p, "abstract_type", None) == "path"
                    or getattr(p, "port_role", None) in ("source_data", "model_sink")
                    or getattr(p, "derived_role", None) in ("source_data", "model_sink")
                    or getattr(p.signature, "abstract_type", None) == "path"
                    for p in cand.inputs.values()
                )
                path_bonus = 6.0 if (file_literals and is_path_consumer and cand_stage == 3) else 0.0
                return (rel * 10.0) + (aff * 5.0) + tok_bonus + path_bonus

            # Limit candidates presented to LLM to top 5 by score
            valid_next.sort(key=_score_next, reverse=True)
            top_valid = valid_next[:5]

            selected_cell: Optional[Cell] = None

            if has_llm:
                try:
                    cands_summary = "\n".join(
                        f"- {c.cell_id}: stage {c.stage}, domain {c.domain_name}"
                        for c in top_valid
                    )
                    sys_prompt = "You are a precise dataflow pipeline router. Output ONLY the exact cell ID of the next step, or FINISH."
                    user_prompt = (
                        f"Target Intent: {prompt}\n"
                        f"Current Pipeline: {[c.cell_id for c in path]}\n"
                        f"Available Next Cells:\n{cands_summary}\n"
                        f"Select the single next CELL_ID:"
                    )
                    resp = mm.generate_text(user_prompt, max_tokens=64, system_prompt=sys_prompt).strip()
                    # Exact-token match only: a candidate id mentioned inside
                    # another token (prose, prefixes) must not win by substring.
                    resp_tokens = set(re.findall(r"[A-Za-z0-9_]+", resp.lower()))
                    for c in top_valid:
                        if c.cell_id.lower() in resp_tokens:
                            selected_cell = c
                            break
                    if "finish" in resp_tokens:
                        break
                except (RuntimeError, ValueError, OSError) as e:
                    logger.warning(f"[M4] LLM stepwise selection failed: {e}")
                    selected_cell = None

            # Structural fallback if LLM is inactive, failed, or unparseable
            if not selected_cell:
                selected_cell = top_valid[0]

            path.append(selected_cell)
            visited_ids.add(selected_cell.cell_id)

            # 3. Coverage Stall Termination: terminate if coverage does not increase for 2 consecutive steps
            new_cov = _covered_clauses_count(path)
            if new_cov > prev_cov:
                prev_cov = new_cov
                stall_count = 0
            else:
                stall_count += 1
                if stall_count >= 2:
                    break

        return path
