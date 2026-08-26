"""
src/planner.py - Neuro-Symbolic Topological Lattice (NSTL)
Translates user natural language requests into structured MacroCell DAGs.
"""

from __future__ import annotations
import json
import re
import numpy as np
from typing import Dict, Any, Optional, List
from log_config import get_logger
from lattice import LatticeOrchestrator, MicroCell, MacroCell, AlgebraicSignature, PortSignature
from inference import ModelManager

logger = get_logger("planner")


class ZeroShotPlanner:
    def __init__(self, orchestrator: LatticeOrchestrator, rag_engine=None):
        self.orchestrator = orchestrator
        self.rag_engine = rag_engine

    def _get_relevant_context(self, prompt: str) -> str:
        if self.rag_engine:
            return self.rag_engine.get_relevant_context(prompt, top_k=25)

        cells = [c for c in self.orchestrator.loaded_cells.values() if isinstance(c, MicroCell)]
        lines = []
        for c in cells[:25]:
            lines.append(f"- ID: {c.cell_id} | In: {c.primary_input} -> Out: {c.primary_output}")
        return "\n".join(lines)

    def run_planning_pass(self, prompt: str) -> Dict[str, Any]:
        model_mgr = ModelManager.get_instance()

        if not model_mgr.can_synthesize():
            logger.info("[PLANNER] LLM disabled; using deterministic typestate planner.")
            return self._run_deterministic_planner(prompt)

        context_nodes = self._get_relevant_context(prompt)

        system_prompt = f"""You are a Software Architect. Decompose the user request into a strict sequence of verified computational node IDs.
Output ONLY valid JSON matching the schema below. No markdown formatting, no commentary.

Schema:
{{
  "cells": [
    {{
      "cell_id": "macro_dynamic_task",
      "type": "macro",
      "stage": 1,
      "sub_cells": ["<node_id_1>", "<node_id_2>", "<node_id_3>"]
    }}
  ]
}}

Available Verified Micro-Nodes:
{context_nodes}

RULES:
1. Select exclusively from the Available Verified Micro-Nodes when a suitable node exists.
2. For end-to-end I/O pipelines (e.g. read CSV/image, transform, save to file), you MUST include the full sequence: Source Node -> Processing Nodes -> Sink/Export Node.
3. If an essential step has no matching node, invent a node ID prefixed with 'SYNTH_'."""

        full_prompt = f"{system_prompt}\n\nUser Request: {prompt}"

        try:
            raw_output = model_mgr.generate_text(full_prompt, max_tokens=3072)
        except Exception as e:
            logger.error(f"[PLANNER ERROR] LLM generation failed: {e}")
            return self._run_deterministic_planner(prompt)

        clean_json = raw_output.strip()
        if clean_json.startswith("```"):
            clean_json = clean_json.split("\n", 1)[-1]
        if clean_json.endswith("```"):
            clean_json = clean_json.rsplit("```", 1)[0]
        clean_json = clean_json.strip()

        try:
            macro_data = json.loads(clean_json)
            self._validate_and_register_plan(macro_data)
            return macro_data
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"[PLANNER] Invalid JSON from LLM ({e}). Falling back to deterministic planner.")
            return self._run_deterministic_planner(prompt)

    def _validate_and_register_plan(self, macro_data: Dict[str, Any]):
        cells = macro_data.get("cells", [])
        for cell_dict in cells:
            sub_cells = cell_dict.get("sub_cells", [])
            for i, sub_id in enumerate(sub_cells):
                if sub_id not in self.orchestrator.loaded_cells:
                    if self.rag_engine and hasattr(self.rag_engine, "index") and self.rag_engine.index is not None:
                        clean_query = sub_id.replace("SYNTH_", "").replace("_", " ")
                        raw_emb = np.array([ModelManager.get_instance().get_embedding(clean_query)], dtype=np.float32)
                        norm = np.linalg.norm(raw_emb)
                        if norm > 0:
                            raw_emb = raw_emb / norm
                            dists, indices = self.rag_engine.index.search(raw_emb, k=1)
                            if len(indices[0]) > 0 and indices[0][0] != -1:
                                match_schema = self.rag_engine.id_to_schema.get(indices[0][0], {})
                                match_id = match_schema.get("cell_id")
                                sim = float(dists[0][0])
                                if match_id and sim >= 0.70 and match_id in self.orchestrator.loaded_cells:
                                    matched_cell = self.orchestrator.loaded_cells[match_id]
                                    # Prevent circular macro references
                                    if isinstance(matched_cell, MacroCell):
                                        continue
                                    logger.info(f"[PLANNER GROUNDING] Mapped '{sub_id}' -> '{match_id}' (sim={sim:.3f})")
                                    sub_cells[i] = match_id
                                    continue

                    if not sub_id.startswith("SYNTH_"):
                        sub_cells[i] = f"SYNTH_{sub_id.upper()}"

            in_sig = AlgebraicSignature("str", "source_identifier")
            out_sig = AlgebraicSignature("any", "any")
            macro_cell = MacroCell(
                cell_id=cell_dict.get("cell_id", "macro_dynamic"),
                stage=1,
                keywords=set(),
                inputs={"input_data": PortSignature("input_data", in_sig)},
                outputs={"output_data": PortSignature("output_data", out_sig)},
                sub_cells=sub_cells,
                domain_name=self.orchestrator.active_domain
            )
            self.orchestrator.inject_cell(macro_cell)

    def _run_deterministic_planner(self, prompt: str) -> Dict[str, Any]:
        tokens = set(re.findall(r"[a-zA-Z_]+", prompt.lower()))
        available_micro = [
            c for c in self.orchestrator.loaded_cells.values()
            if isinstance(c, MicroCell)
            and c.code_template
            and c.code_template.strip()
        ]

        curr_sig = AlgebraicSignature("str", "source_identifier")
        selected_ids: List[str] = []
        used_ids = set()

        for _ in range(6):
            compatible = [
                c for c in available_micro
                if c.cell_id not in used_ids and curr_sig.unifies_with(c.primary_input)
            ]
            if not compatible:
                break

            best_cell = None
            best_score = -1
            for c in compatible:
                overlap = len(tokens.intersection({k.lower() for k in c.keywords}))
                id_tokens = {p.lower() for p in c.cell_id.split("_")}
                overlap += len(tokens.intersection(id_tokens))

                # Encourage natural pipeline progression: Source -> Transform -> Sink
                stage_bonus = 0
                if not selected_ids:
                    if c.stage == 1:
                        stage_bonus = 2
                else:
                    last_stage = self.orchestrator.loaded_cells[selected_ids[-1]].stage
                    if c.stage == last_stage:
                        stage_bonus = 1
                    elif c.stage == last_stage + 1:
                        stage_bonus = 2
                    elif c.stage < last_stage:
                        stage_bonus = -1

                total_score = overlap + stage_bonus

                if total_score > best_score:
                    best_score = total_score
                    best_cell = c

            if not best_cell or best_score <= 0:
                break

            selected_ids.append(best_cell.cell_id)
            used_ids.add(best_cell.cell_id)
            curr_sig = best_cell.primary_output

            if curr_sig.type_name in ("None", "NoneType"):
                break

        if not selected_ids:
            raise ValueError(f"Deterministic planner could not resolve an executable path for: '{prompt}'")

        return {
            "cells": [
                {
                    "cell_id": "macro_deterministic_plan",
                    "type": "macro",
                    "stage": 1,
                    "sub_cells": selected_ids
                }
            ]
        }
