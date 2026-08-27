"""
src/planner.py - Zero-Shot Planner with algorithmic-task bypass and robust JSON.
"""

import json
import re
from typing import Any, Dict, List, Optional, Set

from log_config import get_logger
from lattice import LatticeOrchestrator, Cell, MacroCell, MicroCell, AlgebraicSignature

logger = get_logger('planner')


class ZeroShotPlanner:
    def __init__(self, orchestrator: LatticeOrchestrator, rag: Any):
        self.orchestrator = orchestrator
        self.rag = rag

    def run_planning_pass(self, prompt: str, profile: str = "C") -> Dict[str, Any]:
        context = self.rag.get_relevant_context(prompt, top_k=60)

        # Check for algorithmic seeds via RAG domain / macro detection
        if context:
            for entry in context[:3]:
                if isinstance(entry, dict):
                    cid = entry.get("cell_id", "")
                    dom = (entry.get("domain") or "").lower()
                    score = float(entry.get("score", 0.0))
                    cell = self.orchestrator.loaded_cells.get(cid)
                    if (dom in ("algorithms", "algorithm") or (cell and isinstance(cell, MacroCell) and cell.algorithmic_steps)) and score > 0.35:
                        logger.info(f"[PLANNER] Algorithmic seed found via RAG: {cid}")
                        return {
                            "cells": [{
                                "cell_id": "macro_algo_seeded",
                                "type": "macro",
                                "stage": 2,
                                "sub_cells": [cid]
                            }]
                        }

        context_str = self._format_context(context)

        system_prompt = self._build_system_prompt(context_str)
        user_prompt = f"User Request: {prompt}\n\nOutput ONLY valid JSON."

        raw = None
        try:
            from inference import ModelManager
            mm = ModelManager.get_instance()
            if mm.can_synthesize():
                raw = mm.generate_text(system_prompt + "\n" + user_prompt, max_tokens=1024)
        except Exception as e:
            logger.warning(f"[PLANNER] LLM generation unavailable: {e}")

        parsed = self._safe_parse_json(raw) if raw else None
        if parsed is None:
            logger.warning("[PLANNER] Invalid or missing JSON from LLM. Falling back to deterministic planner.")
            return self._deterministic_fallback(prompt, context)

        for cell_block in parsed.get("cells", []):
            grounded = []
            for sub in cell_block.get("sub_cells", []):
                existing = self._find_closest_existing_cell(sub)
                if existing:
                    grounded.append(existing)
                else:
                    grounded.append(sub)
            cell_block["sub_cells"] = grounded

        return parsed

    def _build_system_prompt(self, context_str: str) -> str:
        return f"""You are a Software Architect. Decompose the user request into a strict sequence of verified computational node IDs.
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
{context_str}

RULES:
1. Select exclusively from the Available Verified Micro-Nodes when a suitable node exists.
2. If an essential step has no matching node, invent a new node ID prefixed with 'SYNTH_' (e.g. 'SYNTH_CALCULATE_METRIC'). The engine will synthesize it at runtime.
3. Order sub_cells sequentially from data source -> transformations -> output/sink.
4. For algorithmic tasks (sorting, graph search, etc.), use a single SYNTH_ node rather than forcing library primitives.
"""

    def _format_context(self, context: List[Any]) -> str:
        lines = []
        for entry in context:
            cid = ""
            if isinstance(entry, dict):
                cid = entry.get("cell_id", "UNKNOWN")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 1:
                cid = str(entry[0])
            elif isinstance(entry, str):
                m = re.search(r"ID:\s*([A-Z0-9_]+)", entry)
                cid = m.group(1) if m else "UNKNOWN"

            cell = self.orchestrator.loaded_cells.get(cid)
            if not cell:
                continue
            in_sig = f"{cell.primary_input.type_name}[{cell.primary_input.state}]"
            out_sig = f"{cell.primary_output.type_name}[{cell.primary_output.state}]"
            lines.append(f"- ID: {cid} | In: {in_sig} -> Out: {out_sig} | Domain: {cell.domain_name}")
        return "\n".join(lines)

    def _safe_parse_json(self, raw: str) -> Optional[Dict[str, Any]]:
        if not raw:
            return None
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
        return None

    def _find_best_match_for_stage(
        self,
        stage: int,
        current_sig: AlgebraicSignature,
        context: List[Any],
        keywords: Set[str],
        goal_domain: Optional[str] = None,
        exclude: Optional[Set[str]] = None
    ) -> Optional[str]:
        exclude = exclude or set()
        best_match = None
        best_score = -1.0
        for entry in context:
            cid = ""
            score = 0.0
            if isinstance(entry, dict):
                cid = entry.get("cell_id", "")
                score = float(entry.get("score", 0.0))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 1:
                cid = str(entry[0])
                score = float(entry[1]) if len(entry) > 1 else 0.0
            elif isinstance(entry, str):
                m = re.search(r"ID:\s*([A-Z0-9_]+)", entry)
                cid = m.group(1) if m else ""

            if not cid or cid in exclude:
                continue

            cell = self.orchestrator.loaded_cells.get(cid)
            if not cell or cell.stage != stage:
                continue

            if not current_sig.unifies_with(cell.primary_input) and not cell.can_accept(current_sig):
                continue

            cell_kws = set(k.lower() for k in cell.keywords)
            overlap = len(keywords & cell_kws)
            if stage == 2 and keywords and overlap == 0:
                continue

            score += overlap * 0.35

            if goal_domain:
                if (cell.domain_name or "").lower() == goal_domain.lower():
                    score *= 1.3
                elif (cell.domain_name or "").lower() not in ("generic", "python_core", "macro"):
                    score *= 0.4

            # Penalize noise helper nodes
            if any(p in cid.lower() for p in ["_group_", "_internal_", "typing_", "withmetadata", "default"]):
                score *= 0.7

            if score > best_score:
                best_score = score
                best_match = cid

        if best_match is None:
            # Fallback to O(1) typed lattice successors directly
            successors = self.orchestrator.get_successors_for_sig(current_sig)
            for cell in successors:
                if cell.cell_id in exclude or cell.stage != stage:
                    continue
                cell_kws = set(k.lower() for k in cell.keywords)
                overlap = len(keywords & cell_kws)
                if stage == 2 and keywords and overlap == 0:
                    continue

                score = 0.5 + overlap * 0.4
                if goal_domain:
                    if (cell.domain_name or "").lower() == goal_domain.lower():
                        score *= 1.3
                    elif (cell.domain_name or "").lower() not in ("generic", "python_core", "macro"):
                        score *= 0.4
                if any(p in cell.cell_id.lower() for p in ["_group_", "_internal_", "typing_", "withmetadata", "default"]):
                    score *= 0.6
                if score > best_score:
                    best_score = score
                    best_match = cell.cell_id

        return best_match

    GENERIC_STOP_WORDS = {
        "and", "then", "to", "with", "from", "the", "a", "an", "in", "on", "of", "for",
        "is", "it", "this", "that", "values", "data", "file", "dataframe", "image",
        "load", "save", "input", "output", "read", "write"
    }

    def _deterministic_fallback(self, prompt: str, context: List[Any]) -> Dict[str, Any]:
        prompt_lower = prompt.lower()
        all_tokens = set(re.findall(r"[a-zA-Z_]+", prompt_lower))
        meaningful_keywords = all_tokens - self.GENERIC_STOP_WORDS

        goal_domain = None
        if context:
            for c in context:
                if isinstance(c, dict) and c.get("domain") and c.get("domain") not in ("generic", "python_core", "macro"):
                    goal_domain = c.get("domain")
                    break

        path: List[str] = []
        current_sig = AlgebraicSignature("str", "source_identifier")

        # Stage 1: Source Loader (uses all_tokens to match read/load)
        stage1_match = self._find_best_match_for_stage(1, current_sig, context, all_tokens, goal_domain)
        if stage1_match:
            path.append(stage1_match)
            current_sig = self.orchestrator.loaded_cells[stage1_match].primary_output
            cell1 = self.orchestrator.loaded_cells[stage1_match]
            meaningful_keywords -= set(k.lower() for k in cell1.keywords)

        # Stage 2: Transformations (consume remaining action keywords)
        while True:
            stage2_match = self._find_best_match_for_stage(2, current_sig, context, meaningful_keywords, goal_domain, exclude=set(path))
            if not stage2_match:
                break
            cell = self.orchestrator.loaded_cells[stage2_match]
            cell_kws = set(k.lower() for k in cell.keywords)
            matched = meaningful_keywords & cell_kws
            if len(matched) == 0 and len(path) > 1:
                break
            path.append(stage2_match)
            current_sig = cell.primary_output
            meaningful_keywords -= matched
            if not meaningful_keywords or len(path) >= 6:
                break

        # Stage 3: Sink / Export
        stage3_match = self._find_best_match_for_stage(3, current_sig, context, all_tokens, goal_domain, exclude=set(path))
        if stage3_match:
            path.append(stage3_match)

        return {
            "cells": [{
                "cell_id": "macro_fallback",
                "type": "macro",
                "stage": 1,
                "sub_cells": path
            }]
        }

    def _find_closest_existing_cell(self, cell_id: str) -> Optional[str]:
        if cell_id in self.orchestrator.loaded_cells:
            return cell_id
        cid_upper = cell_id.upper()
        for cid in self.orchestrator.loaded_cells:
            if cid_upper in cid or cid in cid_upper:
                return cid
        return None

    @staticmethod
    def _slugify(text: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]+", "_", text).strip("_").upper()
