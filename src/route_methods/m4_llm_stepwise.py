from __future__ import annotations
from typing import Dict, List, Optional, Set, Any
from .base import RouteMethod
from lattice import Cell, LatticeOrchestrator, TypeRegistry
from tokenizer import CellTokenizer
from unification import ExecutionContext
from inference import ModelManager
from planner import _is_terminal_sink_cell
try:
    from utils import tokenize_alphanumeric
except ImportError:
    from ..utils import tokenize_alphanumeric


def _ir_filter(candidates, ir_step):
    """(apply_fixes_v7) Constrain candidates to those matching an IR step."""
    if not ir_step: return candidates
    op  = str(ir_step.get("op", "")).strip().lower()
    lib = str(ir_step.get("library", "")).strip().lower()
    if not op and not lib: return candidates
    matched = []
    for c in candidates:
        cid = c.cell_id.lower()
        dom = (getattr(c, "domain_name", "") or "").lower()
        toks = set(str(k).lower() for k in getattr(c, "keywords", []) or [])
        toks |= set(str(k).lower() for k in getattr(c, "semantic_tags", []) or [])
        op_hit = (not op) or (op in cid) or any(op in t for t in toks)
        lib_hit = (not lib) or (lib in dom) or (lib in cid)
        if op_hit and lib_hit: matched.append(c)
    return matched or candidates


class M4LLMStepwiseRouteMethod(RouteMethod):
    name = "m4_llm_stepwise"

    def plan(self, prompt, tunnel, relevance_map, orchestrator=None, ctx=None,
             start_sig=None, goal_sig=None, max_transforms=6, **kwargs):
        orch = orchestrator or self.orchestrator
        if not tunnel: return []
        candidates = [
            c for c in tunnel
            if getattr(c, "node_type", "") not in ("constant", "macro")
            and not c.cell_id.startswith("MACRO_")
            and not (getattr(c, "is_combinator", False) or getattr(c, "node_role", "") == "combinator" or getattr(c, "role", "") == "combinator" or getattr(c, "node_type", "") == "combinator")
        ] or [tunnel[0]]
        mm = ModelManager.get_instance()
        has_llm = mm.profile is not None and getattr(mm.profile, "llm", None) is not None

        ir_steps = []
        if has_llm:
            try:
                from ir_compiler import IRCompiler
                compiler = getattr(self, "_cached_ir", None) or IRCompiler(orch)
                self._cached_ir = compiler
                ir = compiler.compile(prompt)
                if ir is not None: ir_steps = list(ir.steps)
            except Exception: ir_steps = []

        prompt_tokens = CellTokenizer.tokenize_prompt(prompt)
        src_file_literals, dest_file_literals = self.extract_file_literals(prompt or "", ctx=ctx)
        def _score_m4_entry(c):
            sc = relevance_map.get(c.cell_id, 0.0) * 5.0 + len(prompt_tokens & getattr(c, "identity_tokens", c.token_set)) * 3.0
            is_pc = any(getattr(p, "abstract_type", None) == "path"
                        or getattr(p, "port_role", None) in ("source_data", "model_sink")
                        or getattr(p.signature, "abstract_type", None) == "path"
                        for p in c.inputs.values())
            if src_file_literals and is_pc:
                if self.is_file_format_compatible(c, str(src_file_literals[0])):
                    sc += 6.0
                else:
                    sc -= 20.0
            return sc

        entry_pool = [c for c in candidates if getattr(c, "stage", None) == 1] or candidates[:5]
        best_entry = max(entry_pool, key=_score_m4_entry)

        path, visited = [best_entry], {best_entry.cell_id}
        # Prefer the model-compiled IR steps as the clause set -- they came
        # from a model that actually read the prompt, not a punctuation
        # split of it. Fall back to the regex/vocabulary segmenter only when
        # no LLM is loaded or IR compilation produced nothing usable.
        if ir_steps:
            clauses = [
                " ".join(
                    w for w in (
                        str(s.get("op", "")).strip(),
                        str(s.get("library", "")).strip(),
                    ) if w
                )
                for s in ir_steps
            ] or self.segment_prompt_clauses(prompt)
        else:
            clauses = self.segment_prompt_clauses(prompt)
        max_steps = max(2, min(16, max(max_transforms + 2, (len(clauses) if clauses else 1) + 3)))

        has_reg = bool(prompt_tokens & {"regression", "regressor", "continuous"}) and not bool(prompt_tokens & {"classification", "classifier"})
        has_cls = bool(prompt_tokens & {"classification", "classifier"}) and not bool(prompt_tokens & {"regression", "regressor"})

        def _cov_cnt(pc):
            return sum(1 for cl in clauses
                       if any(len(CellTokenizer.tokenize_prompt(cl) & getattr(c, "identity_tokens", c.token_set)) > 0
                              for c in pc))
        prev_cov, stall = _cov_cnt(path), 0

        for step_idx in range(max_steps):
            curr = path[-1]
            if len(path) > 1 and (_is_terminal_sink_cell(curr) or getattr(curr, "stage", None) == 3):
                if step_idx >= max(1, len(clauses) - 1) or _cov_cnt(path) >= len(clauses) * 0.75:
                    break
            curr_out_st = str(getattr(curr.primary_output, "state", "")).lower()
            is_pred = bool(TypeRegistry.get_instance().get_state_properties(curr_out_st).get("is_prediction")) or curr_out_st.startswith("predicted_")

            drop_reasons: Dict[str, int] = {}
            valid = []
            for c in candidates:
                if c.cell_id in visited:
                    drop_reasons["already_visited"] = drop_reasons.get("already_visited", 0) + 1
                    continue
                if not (self.step_unifies(curr, c, prev_path=path) or self.step_unifies_dag(c, path, ctx=ctx)):
                    drop_reasons["unification_failure"] = drop_reasons.get("unification_failure", 0) + 1
                    continue
                c_out = str(getattr(c.primary_output, "state", "")).lower()
                if has_reg and ("label" in c_out or c_out == "predicted_labels"):
                    drop_reasons["reg_label_conflict"] = drop_reasons.get("reg_label_conflict", 0) + 1
                    continue
                if has_cls and c_out == "predicted_values":
                    drop_reasons["cls_value_conflict"] = drop_reasons.get("cls_value_conflict", 0) + 1
                    continue
                if is_pred and (getattr(c, "node_role", "") in ("estimator", "model") or "fit" in c.cell_id.lower()):
                    drop_reasons["post_pred_estimator"] = drop_reasons.get("post_pred_estimator", 0) + 1
                    continue
                valid.append(c)
            import logging
            logger = logging.getLogger(__name__)
            logger.debug("M4 step %d: candidate count=%d, valid count=%d, drop reasons: %s", step_idx, len(candidates), len(valid), drop_reasons)
            if not valid: break

            if ir_steps and step_idx < len(ir_steps):
                filtered = _ir_filter(valid, ir_steps[step_idx])
                if filtered: valid = filtered

            def _score_cand(c: Cell) -> float:
                sc = (
                    relevance_map.get(c.cell_id, 0.0) * 10.0
                    + self.calculate_edge_affinity(curr, c, orch, relevance_map=relevance_map) * 5.0
                    + len(c.token_set & prompt_tokens) * 3.0
                )
                if not dest_file_literals and getattr(c, "stage", None) == 3:
                    is_pc = any(
                        getattr(p, "abstract_type", None) == "path"
                        or getattr(p, "port_role", None) in ("source_data", "model_sink")
                        or getattr(getattr(p, "signature", None), "abstract_type", None) == "path"
                        for p in c.inputs.values()
                    )
                    if is_pc:
                        sc -= 25.0
                return sc

            valid.sort(key=_score_cand, reverse=True)
            top_valid = valid[:5]
            selected = None
            if has_llm:
                try:
                    summary = "\n".join(f"- {c.cell_id}" for c in top_valid)
                    ir_hint = ""
                    if ir_steps and step_idx < len(ir_steps):
                        s = ir_steps[step_idx]
                        ir_hint = f"\nTarget op: {s.get('op','')} (library: {s.get('library','')})"
                    resp = mm.generate_text(
                        f"Intent: {prompt}\nPipeline: {[c.cell_id for c in path]}{ir_hint}\nCandidates:\n{summary}\nSelect next CELL_ID:",
                        max_tokens=64,
                        system_prompt="Output ONLY the exact cell ID from the candidate list, or FINISH.",
                    ).strip()
                    toks = set(tokenize_alphanumeric(resp))
                    if "finish" in toks: break
                    selected = next((c for c in top_valid if c.cell_id.lower() in toks), None)
                except Exception:
                    selected = None
            selected = selected or top_valid[0]
            path.append(selected); visited.add(selected.cell_id)
            new_cov = _cov_cnt(path)
            if new_cov > prev_cov: prev_cov, stall = new_cov, 0
            else:
                stall += 1
                if stall >= 3 and step_idx >= max(1, len(clauses) - 1): break

        # Check fallback condition: len(path) < 5 or clause coverage < coverage_floor_fraction
        try:
            from config import NSTLSettings
            cov_floor = NSTLSettings().coverage_floor_fraction
        except Exception:
            cov_floor = 0.85
        total_clauses = len(clauses) if clauses else 1
        curr_coverage = _cov_cnt(path) / total_clauses

        if len(path) < 5 or curr_coverage < cov_floor:
            import logging
            logger = logging.getLogger(__name__)
            logger.info("M4 fallback triggered: len(path)=%d < 5 or coverage=%.2f < %.2f. Falling back to M1ClauseAnchorRouteMethod.", len(path), curr_coverage, cov_floor)
            try:
                from .m1_clause_anchor import M1ClauseAnchorRouteMethod
                m1 = M1ClauseAnchorRouteMethod(self.orchestrator)
                fallback_path = m1.plan(
                    prompt=prompt,
                    tunnel=tunnel,
                    relevance_map=relevance_map,
                    orchestrator=orch,
                    ctx=ctx,
                    start_sig=start_sig,
                    goal_sig=goal_sig,
                    max_transforms=max_transforms,
                    **kwargs,
                )
                if fallback_path and len(fallback_path) >= len(path):
                    return fallback_path
            except Exception as e:
                logger.warning("M4 fallback to M1 failed: %s", e)

        return path
