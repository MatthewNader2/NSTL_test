"""
src/router.py - Neuro-Symbolic Topological Lattice (NSTL)
FIXED: Domain-aware scoring, keyword overlap, algorithmic bypass, robust RAG format handling,
       and preserves original return type contract: (List[Cell], Set[str]).
"""

from __future__ import annotations
import math
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple, Any

import numpy as np
import torch
from log_config import get_logger
import heapq
from lattice import LatticeOrchestrator, Cell, MicroCell, MacroCell, AlgebraicSignature, PortSignature
from inference import ModelManager

logger = get_logger('router')


def extract_file_paths_and_extensions(text: str) -> Tuple[List[str], List[str]]:
    """
    Dynamically extracts file path literals and their trailing format tokens in chronological order.
    Ignores numeric floats (e.g. 3.14) and module/attribute accesses (e.g. cv2.imread, pd.DataFrame).
    """
    pattern = r'(?:[\'"])([^\'"\s]+\.([a-zA-Z0-9_]+))(?:[\'"])|(?:\b|\B(?=[/.]))([a-zA-Z0-9_\-./\\]+\.([a-zA-Z0-9_]+))'
    ignore_prefixes = {"cv2", "np", "pd", "plt", "scipy", "os", "sys", "torch", "tf", "sklearn", "skimage", "sns", "df", "self", "model"}

    paths = []
    exts = []
    for m in re.finditer(pattern, text):
        if m.group(1):
            full_path = m.group(1).strip()
            ext = m.group(2).strip().lower()
        else:
            full_path = m.group(3).strip()
            ext = m.group(4).strip().lower()

        if ext.isdigit() or len(ext) > 10:
            continue
        # If followed by '(', it is a function call, not a file
        end_idx = m.end()
        if end_idx < len(text) and text[end_idx] == '(':
            continue
        prefix = full_path.split('.')[0].lower()
        if prefix in ignore_prefixes:
            continue
        raw_ext = full_path.split('.')[-1]
        # Ignore CamelCase / PascalCase attribute/class access like pd.DataFrame
        if raw_ext[0].isupper() and any(c.islower() for c in raw_ext):
            continue

        if full_path not in paths:
            paths.append(full_path)
            exts.append(ext)

    return paths, exts


def log_coverage_gap(prompt: str, domain_guess: str, score: float, node_id: str = "") -> None:
    os.makedirs("logs", exist_ok=True)
    with open(os.path.join("logs", "coverage_gaps.log"), "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "prompt": prompt,
            "domain_guess": domain_guess,
            "score": score,
            "node_id": node_id,
            "timestamp": time.time()
        }) + "\n")


@dataclass(frozen=True)
class SemanticFrame:
    """
    Immutable pre-compiled semantic representation of the input prompt.
    Parsed ONCE upfront in Phase 1; queried via O(1) attribute and set lookups in Phase 2 traversal.
    """
    raw_prompt: str
    clauses: Tuple[str, ...]
    clause_intents: Tuple[FrozenSet[str], ...]
    has_explicit_sink: bool

    # Classified Token Pools
    partitioning_entities: FrozenSet[str]  # Structural grouping/splitting identifiers (e.g. 'department')
    operand_entities: FrozenSet[str]       # Computational/transformation targets (e.g. 'sales')
    ordering_keys: FrozenSet[str]          # Explicit sorting keys (e.g. 'salary', 'age')
    ascending: Optional[bool]              # True if ascending, False if descending, None if unspecified
    sort_by_metric: bool                   # True if sorting by metric/values rather than named keys
    literal_constants: FrozenSet[str]      # Scalar values, numerical bounds, explicit booleans (e.g. '0', 'None')

    # Dynamic Format Tokenization & Chronological Paths
    path_literals: Tuple[str, ...] = ()
    format_tokens: Tuple[str, ...] = ()

    # Pre-computed per-clause flags for O(1) step cost evaluation
    clause_has_literal: Tuple[bool, ...] = ()
    clause_has_reduction: Tuple[bool, ...] = ()

    @classmethod
    def build(cls, prompt: str, astar_instance: Optional[Any] = None) -> "SemanticFrame":
        prompt_lower = prompt.lower()

        # 1. Clause segmentation
        raw_clauses = [
            c.strip() for c in re.split(r"[,;]|\band\b|\bthen\b|\bwith\b|\bafter\b", prompt, flags=re.IGNORECASE)
            if len(c.strip()) >= 3
        ]
        clauses = tuple(raw_clauses) if raw_clauses else (prompt.strip(),)

        # 2. Per-clause intent extraction & flags
        clause_intents_list = []
        clause_has_literal_list = []
        clause_has_reduction_list = []

        for cl in clauses:
            cl_lower = cl.lower()
            if astar_instance and hasattr(astar_instance, "extract_required_intents"):
                c_intents = frozenset(astar_instance.extract_required_intents(cl))
            else:
                c_intents = frozenset(t for t in re.findall(r"[a-zA-Z0-9_]+", cl_lower) if len(t) >= 3 and not t.isdigit())
            clause_intents_list.append(c_intents)

            # Literal detection: scalar numbers, none, null, nan
            has_lit = bool(re.search(r"\b(?:with\s+)?(?:0|none|nan|null|\d+)\b", cl_lower))
            clause_has_literal_list.append(has_lit)

            # Reduction detection: mean, average, median, mode
            has_red = any(r in cl_lower for r in ("mean", "average", "avg", "median", "mode"))
            clause_has_reduction_list.append(has_red)

        COLUMN_STOP_WORDS = {
            "descending", "ascending", "the", "a", "an", "column", "columns",
            "data", "file", "csv", "by", "sort", "order", "and", "or", "in", "of",
            "to", "from", "with", "into", "as", "values", "value", "dataset", "table",
            "records", "rows", "row", "asc", "desc", "true", "false", "none", "null",
            "sum", "total", "mean", "average", "avg", "count", "min", "max", "std", "var",
            "all", "any", "each", "every", "index", "on", "calculate", "compute", "find",
            "get", "determine", "apply", "output", "input"
        }

        # Partitioning entities (e.g. group by department)
        part_entities = []
        for m in re.finditer(
            r"(?:group\s+by|grouped\s+by|per|for\s+each)\s+(?:the\s+|a\s+|an\s+|column\s+)*[\'\"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'\"]?",
            prompt,
            re.IGNORECASE
        ):
            val = m.group(1).strip("\"'")
            if val.lower() not in COLUMN_STOP_WORDS:
                part_entities.append(val)

        # Ordering keys (e.g. sort by salary, sort values by age)
        ord_keys = []
        for m in re.finditer(
            r"(?:sort\s+(?:values\s+)?by|sorted\s+(?:values\s+)?by|order\s+by)\s+(?:the\s+|a\s+|an\s+|column\s+)*[\'\"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'\"]?",
            prompt,
            re.IGNORECASE
        ):
            val = m.group(1).strip("\"'")
            if val.lower() not in COLUMN_STOP_WORDS:
                ord_keys.append(val)

        # Operand / Measure entities (e.g. total sales sum, calculate total revenue)
        op_entities = []
        for m in re.finditer(
            r"(?:total|sum|mean|average|avg|median|min|max|count|std|var)\s+(?:of\s+)?(?:the\s+)?[\'\"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'\"]?",
            prompt,
            re.IGNORECASE
        ):
            val = m.group(1).strip("\"'")
            if val.lower() not in COLUMN_STOP_WORDS:
                op_entities.append(val)

        for m in re.finditer(
            r"[\'\"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'\"]?\s+(?:sum|total|mean|average|avg|count)",
            prompt,
            re.IGNORECASE
        ):
            val = m.group(1).strip("\"'")
            if val.lower() not in COLUMN_STOP_WORDS:
                op_entities.append(val)

        # Literal constants (e.g. 0, None, numbers)
        literals = set(re.findall(r"\b\d+(?:\.\d+)?\b", prompt))
        for m in re.finditer(r"\bwith\s+(?:the\s+|a\s+|an\s+|value\s+of\s+|values?\s+of\s+|values?\s+)*([\"']?[a-zA-Z0-9_.-]+[\"']?)", prompt, re.IGNORECASE):
            w_val = m.group(1).strip("\"'")
            if w_val.lower() not in COLUMN_STOP_WORDS:
                literals.add(w_val)

        # Directional flags
        ascending = None
        if "ascending" in prompt_lower or "asc" in prompt_lower.split():
            ascending = True
        elif "descending" in prompt_lower or "desc" in prompt_lower.split() or "reverse" in prompt_lower:
            ascending = False

        sort_by_metric = bool(re.search(
            r"\b(?:sort\s+values|sort\s+by\s+values?|sort\s+by\s+total|sort\s+by\s+sum|sort\s+by\s+metric|sort\s+descending|sort\s+ascending)\b",
            prompt_lower
        ))
        if not ord_keys and any(k in prompt_lower for k in ("sort", "order")):
            sort_by_metric = True

        sink_verbs = {"save", "write", "export", "dump", "output", "dest", "sink", "print", "display", "show"}
        has_sink = any(re.search(rf"\b{re.escape(v)}\b", prompt_lower) for v in sink_verbs)

        # Dynamic format and path extraction (no static extension tables or library-specific checks)
        paths, exts = extract_file_paths_and_extensions(prompt)

        return cls(
            raw_prompt=prompt,
            clauses=clauses,
            clause_intents=tuple(clause_intents_list),
            has_explicit_sink=has_sink,
            partitioning_entities=frozenset(part_entities),
            operand_entities=frozenset(op_entities),
            ordering_keys=frozenset(ord_keys),
            ascending=ascending,
            sort_by_metric=sort_by_metric,
            literal_constants=frozenset(literals),
            path_literals=tuple(paths),
            format_tokens=tuple(exts),
            clause_has_literal=tuple(clause_has_literal_list),
            clause_has_reduction=tuple(clause_has_reduction_list)
        )


@dataclass(order=True)
class SemanticSearchNode:
    f_score: float
    g_score: float = field(compare=False)
    current_sig: PortSignature = field(compare=False)
    stage_cursor: int = field(compare=False)
    remaining_intents: Tuple[str, ...] = field(compare=False)
    path: List[Cell] = field(compare=False)


class SemanticStateAStar:
    """
    A* Graph Search over State = (CurrentTypestate, StageCursor, RemainingIntents).
    Implements sequential sub-goal tracking, same-type bridge elimination,
    adapter gating, and dynamic terminal goal conditions.
    """
    STOP_WORDS = {
        "and", "then", "to", "with", "from", "the", "a", "an", "in", "on", "of", "for",
        "is", "it", "this", "that", "values", "data", "file", "after", "by", "into",
        "dataset", "table", "missing", "calculate", "compute", "records", "rows", "row",
        "column", "columns", "field", "fields", "feature", "features", "dataframe", "series", "index",
        "implement", "sample", "example", "perform", "execute", "build", "create"
    }

    FILE_EXTENSIONS = (
        r"csv|tsv|json|parquet|xlsx|jpg|jpeg|png|bmp|webp|tif|tiff|txt|db|sqlite|mat|h5|hdf5|"
        r"pdf|md|py|npy|npz|pkl|pickle|feather|orc|avro|yaml|yml|toml|ini"
    )

    STAGE_ROLE_TAGS = {
        1: {"read", "load", "ingest", "input", "import", "source"},
        3: {"save", "write", "export", "dump", "output", "dest", "sink", "print", "display", "show"}
    }

    def __init__(self, orchestrator: LatticeOrchestrator, rag_engine: Any = None):
        self.orchestrator = orchestrator
        self.rag = rag_engine
        self._format_affinity_cache: Dict[Tuple[str, str], bool] = {}

    def check_format_affinity(self, cell: Cell, format_token: str) -> bool:
        """
        Evaluates dynamic format compatibility via lexical presence across cell metadata/AST,
        falling back to vector cosine similarity via ModelManager dense embeddings.
        """
        if not format_token:
            return True
        fmt = format_token.lower()
        cache_key = (cell.cell_id, fmt)
        if cache_key in self._format_affinity_cache:
            return self._format_affinity_cache[cache_key]

        # 1. Lexical presence check
        cell_id_l = cell.cell_id.lower()
        if fmt in cell_id_l:
            self._format_affinity_cache[cache_key] = True
            return True
        tags = {str(t).lower() for t in getattr(cell, "semantic_tags", []) or []}
        if fmt in tags:
            self._format_affinity_cache[cache_key] = True
            return True
        kws = {str(k).lower() for k in getattr(cell, "keywords", []) or []}
        if fmt in kws:
            self._format_affinity_cache[cache_key] = True
            return True
        doc = (getattr(cell, "docstring", "") or "").lower()
        if fmt in doc:
            self._format_affinity_cache[cache_key] = True
            return True
        code = (getattr(cell, "code_template", "") or "").lower()
        if fmt in code:
            self._format_affinity_cache[cache_key] = True
            return True

        # 2. Vector cosine similarity fallback
        try:
            mm = ModelManager.get_instance()
            if mm and mm.active_profile:
                fmt_emb = np.array(mm.get_embedding(f"file format {fmt}"), dtype=np.float32)
                from internal_rag import build_cell_embedding_text
                cell_text = build_cell_embedding_text(cell)
                cell_emb = np.array(mm.get_embedding(cell_text), dtype=np.float32)
                norm_fmt = float(np.linalg.norm(fmt_emb))
                norm_cell = float(np.linalg.norm(cell_emb))
                if norm_fmt > 0 and norm_cell > 0:
                    sim = float(np.dot(fmt_emb, cell_emb) / (norm_fmt * norm_cell))
                    result = (sim >= 0.45)
                    self._format_affinity_cache[cache_key] = result
                    return result
        except Exception:
            pass

        self._format_affinity_cache[cache_key] = False
        return False

    def heuristic(self, current_sig: PortSignature, goal_sig: Optional[PortSignature], remaining_intents: Tuple[str, ...]) -> float:
        h = len(remaining_intents) * 0.8  # Penalty for unfulfilled sub-intents
        if goal_sig is not None:
            if not current_sig.unifies_with(goal_sig):
                h += 1.5
        return h

    FLAG_MODIFIERS = {"ascending", "descending", "true", "false", "inplace", "axis"}

    ACTION_KEYWORDS = {
        "plot", "save", "load", "read", "write", "mean", "sort", "train",
        "filter", "clean", "cluster", "scale", "blur", "hist", "histogram", "chart"
    }

    def extract_required_intents(self, prompt: str) -> List[str]:
        prompt_lower = prompt.lower()
        file_stems = set(re.findall(rf"([a-zA-Z0-9_-]+)\.(?:{self.FILE_EXTENSIONS})", prompt_lower))
        by_cols = set(re.findall(r"(?:by|column|col)\s+([a-zA-Z0-9_]+)", prompt_lower))
        exclude = (file_stems - self.ACTION_KEYWORDS) | by_cols | self.STOP_WORDS | self.FLAG_MODIFIERS

        tokens = [
            t for t in re.findall(r"[a-zA-Z0-9_]+", prompt_lower)
            if len(t) >= 3 and not t.isdigit()
        ]
        intents = [
            t for t in tokens
            if t in self.ACTION_KEYWORDS or t not in exclude or t in ("csv", "json", "jpg", "png", "image")
        ]
        return list(dict.fromkeys(intents))

    @staticmethod
    def _is_adapter_cell(cell: Cell) -> bool:
        """Dynamically identifies adapter / conversion / constructor nodes without hardcoding names."""
        cid_l = cell.cell_id.lower()
        tags = set(str(t).lower() for t in getattr(cell, "semantic_tags", []))
        kws = set(str(k).lower() for k in getattr(cell, "keywords", []))
        all_desc = tags | kws | {cid_l}
        adapter_terms = {"adapter", "converter", "conversion", "wrapper", "constructor"}
        if any(term in all_desc for term in adapter_terms):
            return True
        for part in cid_l.split("_"):
            if part in ("from", "as", "into", "wrap", "cast", "convert"):
                return True
        return False

    @staticmethod
    def _is_adapter_redundant(
        current_sig: PortSignature,
        candidate_pool: Optional[List[Cell]],
        stage_cursor: int,
        cell_clause_map: Optional[Dict[str, Set[int]]]
    ) -> bool:
        """Returns True if upcoming downstream candidate cells already unify with current_sig."""
        if not candidate_pool:
            return False
        downstream_candidates = []
        for c in candidate_pool:
            c_clauses = cell_clause_map.get(c.cell_id, set()) if cell_clause_map else set()
            if not c_clauses or any(s >= stage_cursor for s in c_clauses):
                downstream_candidates.append(c)
        if not downstream_candidates:
            return False

        for dc in downstream_candidates:
            if not SemanticStateAStar._is_adapter_cell(dc):
                if hasattr(dc, "primary_input") and current_sig.unifies_with(dc.primary_input):
                    return True
        return False

    def search(
        self,
        start_sig: PortSignature,
        goal_sig: Optional[PortSignature],
        required_intents: List[str],
        candidate_pool: Optional[List[Cell]] = None,
        clauses: Optional[List[str]] = None,
        cell_clause_map: Optional[Dict[str, Set[int]]] = None,
        has_explicit_sink: bool = False,
        clause_intents_list: Optional[List[List[str]]] = None,
        frame: Optional[SemanticFrame] = None
    ) -> List[Cell]:
        if frame is not None:
            clauses = list(frame.clauses)
            has_explicit_sink = frame.has_explicit_sink
            if clause_intents_list is None:
                clause_intents_list = [list(ci) for ci in frame.clause_intents]

        initial_intents = tuple(sorted(set(required_intents)))
        open_set: List[SemanticSearchNode] = []
        if start_sig is not None:
            start_node = SemanticSearchNode(
                f_score=self.heuristic(start_sig, goal_sig, initial_intents),
                g_score=0.0,
                current_sig=start_sig,
                stage_cursor=0,
                remaining_intents=initial_intents,
                path=[]
            )
            open_set.append(start_node)

        visited: Set[Tuple[str, str, int, Tuple[str, ...]]] = set()
        best_partial_path: List[Cell] = []
        min_remaining = len(initial_intents) + 1
        pool_ids = {c.cell_id for c in candidate_pool} if candidate_pool else None

        # Pre-compute immutable cell metadata for O(1) attribute and intent lookup during expansion
        # Schema: (c_tags: frozenset, has_reduction: bool, has_lit_slot: bool, is_internal: bool, is_unverified_default: bool)
        cell_meta: Dict[str, Tuple[FrozenSet[str], bool, bool, bool, bool]] = {}
        target_cells = candidate_pool if candidate_pool is not None else self.orchestrator.loaded_cells.values()
        for cell in target_cells:
            c_tags = (
                set(getattr(cell, "semantic_tags", []))
                | set(getattr(cell, "keywords", []))
                | {cell.cell_id.lower(), getattr(cell, "domain_name", "").lower()}
                | {p.type_name.lower() for p in list(cell.inputs.values()) + list(cell.outputs.values()) if hasattr(p, "type_name")}
            )
            if cell.stage == 1:
                c_tags.update(self.STAGE_ROLE_TAGS[1])
            elif getattr(cell, "primary_output", None) and cell.primary_output.state in ("filepath_written", "saved", "exported", "displayed"):
                c_tags.update(self.STAGE_ROLE_TAGS[3])

            has_red = any(r in cell.cell_id.lower() or r in cell.code_template.lower() for r in ("_mean", "_avg", "_median", ".mean(", ".median("))
            has_lit = any(p in getattr(cell, "inputs", {}) for p in ("value", "val", "fill_value", "to_replace"))
            is_int = cell.cell_id.startswith("_") or "_internal_" in cell.cell_id.lower()
            is_unver = "_default" in cell.cell_id.lower() and not getattr(cell, "verified", False)
            cell_meta[cell.cell_id] = (frozenset(c_tags), has_red, has_lit, is_int, is_unver)

        # Dynamic Entry Point Seeding: When start_sig is not fixed or no external file is referenced,
        # seed candidate entry cells with optional/nullary inputs directly into open_set.
        has_file = bool(frame and (frame.path_literals or frame.format_tokens)) if frame else False
        if start_sig is None or not has_file:
            for cell in target_cells:
                if cell.stage == 3 or not all(not p.required for p in cell.inputs.values()):
                    continue
                c_tags, has_red, has_lit, is_int, is_unver = cell_meta.get(cell.cell_id, (frozenset(), False, False, False, False))
                if is_int:
                    continue
                satisfied = set()
                for intent in initial_intents:
                    intent_l = intent.lower()
                    if intent_l in c_tags or (len(intent_l) >= 4 and any(len(tag) >= 4 and (intent_l in tag or tag in intent_l) for tag in c_tags)):
                        satisfied.add(intent)
                if satisfied:
                    new_intents = tuple(i for i in initial_intents if i not in satisfied)
                    step_cost = 0.8 if getattr(cell, "verified", False) else 1.5
                    h_val = self.heuristic(cell.primary_output, goal_sig, new_intents)
                    heapq.heappush(open_set, SemanticSearchNode(
                        f_score=step_cost + h_val,
                        g_score=step_cost,
                        current_sig=cell.primary_output,
                        stage_cursor=0,
                        remaining_intents=new_intents,
                        path=[cell]
                    ))

        if not open_set:
            fallback_port = PortSignature("input_data", AlgebraicSignature("str", "source_identifier"))
            open_set.append(SemanticSearchNode(
                f_score=self.heuristic(fallback_port, goal_sig, initial_intents),
                g_score=0.0,
                current_sig=fallback_port,
                stage_cursor=0,
                remaining_intents=initial_intents,
                path=[]
            ))

        while open_set:
            current = heapq.heappop(open_set)

            # Goal Check: All intents consumed AND valid terminal condition satisfied
            if len(current.remaining_intents) == 0:
                if has_explicit_sink:
                    if current.path and (current.path[-1].stage == 3 or current.current_sig.state in ("filepath_written", "saved", "exported", "displayed")):
                        return current.path
                else:
                    if goal_sig is None or current.current_sig.unifies_with(goal_sig):
                        return current.path
                    if current.current_sig.type_name not in ("filepath", "None", "NoneType"):
                        return current.path

            if len(current.remaining_intents) < min_remaining:
                min_remaining = len(current.remaining_intents)
                best_partial_path = current.path

            state_key = (current.current_sig.type_name, current.current_sig.state, current.stage_cursor, current.remaining_intents)
            if state_key in visited:
                continue
            visited.add(state_key)

            # Expand successors
            if candidate_pool is not None and len(candidate_pool) <= 1000:
                successors = [c for c in candidate_pool if current.current_sig.unifies_with(c.primary_input)]
            else:
                successors = self.orchestrator.get_successors_for_sig(current.current_sig)
                if pool_ids is not None:
                    successors = [c for c in successors if c.cell_id in pool_ids]

            for cell in successors:
                if cell in current.path:
                    continue

                # Pipeline dataflow invariant: upstream output must unify with downstream primary input
                if not current.current_sig.unifies_with(cell.primary_input):
                    continue

                out_sig = cell.primary_output

                # O(1) Bitset Dead-End Reachability Pruning
                if goal_sig is not None and hasattr(self.orchestrator, "can_reach_type"):
                    if not self.orchestrator.can_reach_type(out_sig.type_name, goal_sig.type_name):
                        continue

                # O(1) cell metadata retrieval
                meta = cell_meta.get(cell.cell_id)
                if meta is None:
                    c_tags = (
                        set(getattr(cell, "semantic_tags", []))
                        | set(getattr(cell, "keywords", []))
                        | {cell.cell_id.lower(), getattr(cell, "domain_name", "").lower()}
                        | {p.type_name.lower() for p in list(cell.inputs.values()) + list(cell.outputs.values()) if hasattr(p, "type_name")}
                    )
                    if cell.stage == 1:
                        c_tags.update(self.STAGE_ROLE_TAGS[1])
                    elif getattr(cell, "primary_output", None) and cell.primary_output.state in ("filepath_written", "saved", "exported"):
                        c_tags.update(self.STAGE_ROLE_TAGS[3])
                    has_red = any(r in cell.cell_id.lower() or r in cell.code_template.lower() for r in ("_mean", "_avg", "_median", ".mean(", ".median("))
                    has_lit = any(p in getattr(cell, "inputs", {}) for p in ("value", "val", "fill_value", "to_replace"))
                    is_int = cell.cell_id.startswith("_") or "_internal_" in cell.cell_id.lower()
                    is_unver = "_default" in cell.cell_id.lower() and not getattr(cell, "verified", False)
                    meta = (frozenset(c_tags), has_red, has_lit, is_int, is_unver)
                    cell_meta[cell.cell_id] = meta

                cell_tags, has_reduction, has_lit_slot, is_internal, is_unverified_default = meta

                # Active stage intent tracking
                cur_stage = min(current.stage_cursor, len(clause_intents_list) - 1) if clause_intents_list else 0

                # Fast O(1) set-intersection for intent satisfaction
                satisfied = set()
                for intent in current.remaining_intents:
                    intent_l = intent.lower()
                    if intent_l in cell_tags:
                        satisfied.add(intent)
                    elif len(intent_l) >= 4 and any(len(tag) >= 4 and (intent_l in tag or tag in intent_l) for tag in cell_tags):
                        satisfied.add(intent)

                if cell.stage == 2 and not satisfied:
                    # Once all user intents are satisfied, do not chain unsolicited Stage 2 transformations
                    if len(current.remaining_intents) == 0:
                        continue
                    # Prune zero-progress same-type transformations: no intent satisfied and no type change
                    if cell.primary_input.type_name == cell.primary_output.type_name:
                        continue

                if cell.stage == 3 and not satisfied:
                    # Never chain unsolicited Stage 3 sink cells when intents are not satisfied
                    continue

                # Topological Stage Invariant: Cannot transition from Stage 3 Sink back to Stage 2 Processing
                if current.path and current.path[-1].stage == 3 and cell.stage == 2:
                    continue

                cell_stages = cell_clause_map.get(cell.cell_id, set()) if cell_clause_map else set()
                stage_jump_penalty = 0.0
                next_stage = cur_stage
                is_sink = cell.stage == 3 or (getattr(cell, "primary_output", None) and cell.primary_output.state in ("filepath_written", "saved", "exported"))

                if cell_stages:
                    valid_forward_stages = [s for s in cell_stages if s >= cur_stage]
                    if not valid_forward_stages:
                        if is_sink and has_explicit_sink:
                            # Sink operations mentioned upfront in out-of-order phrasing execute at the terminal stage
                            min_forward = max(cur_stage, len(clauses) - 1) if clauses else cur_stage
                            max_forward = min_forward
                            valid_forward_stages = [min_forward]
                        else:
                            # All stages for this cell are strictly in the past: regressive transition
                            continue
                    min_forward = min(valid_forward_stages)
                    max_forward = max(valid_forward_stages)
                    if min_forward <= cur_stage + 1:
                        # Monotonic progression (same stage or next stage)
                        stage_jump_penalty = 0.0
                        next_stage = max(cur_stage, max_forward)
                    else:
                        # Out-of-order forward jump: exponential heuristic penalty
                        jump = min_forward - cur_stage - 1
                        stage_jump_penalty = 15.0 * (2.0 ** jump)
                        next_stage = max_forward
                else:
                    if cell_clause_map:
                        stage_jump_penalty = 8.0

                # 2. Zero-Progress Same-Type Transformations
                zero_progress_penalty = 0.0
                if cell.primary_input.type_name == cell.primary_output.type_name:
                    if not satisfied:
                        zero_progress_penalty = 10.0

                # 3. Adapter-Gate Rule
                if self._is_adapter_cell(cell):
                    if self._is_adapter_redundant(current.current_sig, candidate_pool, next_stage, cell_clause_map):
                        continue

                new_intents = tuple(
                    intent for intent in current.remaining_intents
                    if intent not in satisfied
                )

                step_cost = (0.8 if getattr(cell, "verified", False) else 1.5) + stage_jump_penalty + zero_progress_penalty
                prio = getattr(cell, "source_priority", 50)
                step_cost += (prio / 50.0)
                if is_internal:
                    step_cost += 10.0
                elif is_unverified_default:
                    step_cost += 6.0

                # Dynamic format alignment penalty (prevents format-incompatible loaders and savers)
                format_penalty = 0.0
                if frame is not None and frame.format_tokens:
                    if len(frame.format_tokens) >= 2:
                        ingress_fmt = frame.format_tokens[0]
                        egress_fmt = frame.format_tokens[-1]
                    elif frame.has_explicit_sink:
                        ingress_fmt = None
                        egress_fmt = frame.format_tokens[0]
                    else:
                        ingress_fmt = frame.format_tokens[0]
                        egress_fmt = None

                    if cell.stage == 1 and ingress_fmt:
                        if not self.check_format_affinity(cell, ingress_fmt):
                            format_penalty = 30.0
                    elif is_sink and egress_fmt:
                        if not self.check_format_affinity(cell, egress_fmt):
                            format_penalty = 30.0

                step_cost += format_penalty

                # Literal Slot Preference Heuristic (Pure O(1) lookup via SemanticFrame)
                if frame is not None and frame.clauses:
                    target_stage = min(next_stage, len(frame.clauses) - 1)
                    if frame.clause_has_literal[target_stage] and not frame.clause_has_reduction[target_stage]:
                        if has_reduction:
                            step_cost += 3.0
                        if has_lit_slot:
                            step_cost -= 0.2
                elif clauses:
                    target_stage = min(next_stage, len(clauses) - 1)
                    t_lower = clauses[target_stage].lower()
                    if any(l in t_lower for l in ("0", "none", "nan", "null")) and not any(r in t_lower for r in ("mean", "average", "avg")):
                        if has_reduction:
                            step_cost += 3.0
                        if has_lit_slot:
                            step_cost -= 0.2

                new_g = current.g_score + step_cost
                new_f = new_g + self.heuristic(out_sig, goal_sig, new_intents)

                heapq.heappush(open_set, SemanticSearchNode(
                    f_score=new_f,
                    g_score=new_g,
                    current_sig=out_sig,
                    stage_cursor=next_stage,
                    remaining_intents=new_intents,
                    path=current.path + [cell]
                ))

        return best_partial_path


class HardwareProfiler:
    _cached_device: Optional[str] = None
    _config: Dict[str, str] = {
        'embedder': 'auto',
        'llm': 'auto',
        'trees': 'ram'
    }

    @classmethod
    def set_config(cls, embedder_device: str = "auto", llm_device: str = "auto", trees_storage: str = "ram"):
        cls._config['embedder'] = (embedder_device or 'auto').lower()
        cls._config['llm'] = (llm_device or 'auto').lower()
        cls._config['trees'] = (trees_storage or 'ram').lower()

    @classmethod
    def get_embedder_device(cls) -> str:
        if cls._config.get('embedder', 'auto') != 'auto':
            return cls._config['embedder']
        return cls.get_optimal_device()

    @classmethod
    def get_llm_device(cls) -> str:
        if cls._config.get('llm', 'auto') != 'auto':
            return cls._config['llm']
        return cls.get_optimal_device()

    @classmethod
    def get_optimal_device(cls) -> str:
        if cls._cached_device is not None:
            return cls._cached_device
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        cls._cached_device = device
        logger.info(f"[HARDWARE PROFILER] Optimal compute device: {device.upper()}")
        return device


class MCTSNode:
    __slots__ = ['cell_id', 'signature', 'parent', 'children', 'visits', 'q_value']

    def __init__(self, cell_id: str, signature: AlgebraicSignature, parent: Optional[MCTSNode] = None):
        self.cell_id = cell_id
        self.signature = signature
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits: int = 0
        self.q_value: float = 0.0

    def ucb1(self, c_param: float = 0.5) -> float:
        if self.visits == 0:
            return float('inf')
        parent_visits = self.parent.visits if self.parent else 1
        return (self.q_value / self.visits) + c_param * math.sqrt(2 * math.log(parent_visits) / self.visits)


class MCTSEngine:
    """Deterministic A* / Best-First typestate bridge engine over the typed lattice."""

    def __init__(self, orchestrator: LatticeOrchestrator):
        self.orchestrator = orchestrator

    def search(
        self,
        start_sig: AlgebraicSignature,
        goal_sig: AlgebraicSignature,
        max_depth: int = 6,
        iterations: int = 300,
        prompt_keywords: Optional[Set[str]] = None
    ) -> List[Cell]:
        """Returns List[Cell] to bridge between start_sig and goal_sig."""
        if start_sig.unifies_with(goal_sig):
            return []

        import heapq
        # Priority queue entries: (cost, depth, counter, current_sig, [cell_ids])
        counter = 0
        pq = [(0.0, 0, counter, start_sig, [])]
        visited_sigs = set()
        kw_set = prompt_keywords or set()

        while pq:
            cost, depth, _, curr_sig, path = heapq.heappop(pq)

            if curr_sig.unifies_with(goal_sig) and path:
                return [self.orchestrator.loaded_cells[cid] for cid in path if cid in self.orchestrator.loaded_cells]

            if depth >= max_depth:
                continue

            sig_key = (curr_sig.type_name, curr_sig.state)
            if sig_key in visited_sigs:
                continue
            visited_sigs.add(sig_key)

            # O(1) indexed lookup of downstream successors
            for cell in self.orchestrator.get_successors_for_sig(curr_sig):
                if cell.cell_id in path:
                    continue

                next_sig = cell.primary_output
                cell_kws = set(k.lower() for k in cell.keywords)
                overlap = len(kw_set & cell_kws) if kw_set else 0

                # Priority: prefer fewer hops, higher keyword overlap, avoid internal helper noise
                step_cost = 1.0 - (0.25 * min(overlap, 3))
                if any(p in cell.cell_id.lower() for p in ["_group_", "_internal_", "typing_"]):
                    step_cost += 0.5

                counter += 1
                heapq.heappush(pq, (cost + step_cost, depth + 1, counter, next_sig, path + [cell.cell_id]))

        return []


def is_goal_satisfied(current_sig: Any, goal_sig: Optional[Any], selected_path: List[Cell]) -> bool:
    """
    Goal satisfaction checking:
    Determines if current typestate satisfies target goal or reaches a terminal export state.
    """
    if goal_sig is not None:
        target_sig = goal_sig.signature if hasattr(goal_sig, "signature") else goal_sig
        c_sig = current_sig.signature if hasattr(current_sig, "signature") else current_sig
        return c_sig.unifies_with(target_sig)
    # If no explicit goal signature is provided, terminate when all intent waypoints are met
    # and the current cell has produced a valid terminal or exported state
    if selected_path and selected_path[-1].stage == 3:
        return True
    return False


class LatticeRouter:
    """
    Semantic router with type-monadic beam search and planner synchronization.
    Supports both (List[Cell], Set[str]) and List[Cell] return formats dynamically.
    """

    def __init__(
        self,
        orchestrator: LatticeOrchestrator,
        rag_engine: Any = None,
        reranker: Optional[Any] = None,
        internal_rag: Optional[Any] = None
    ):
        self.orchestrator = orchestrator
        self.rag = rag_engine if rag_engine is not None else internal_rag
        self.reranker = reranker
        self.mcts = MCTSEngine(orchestrator)
        self._keyword_cache: Dict[str, Set[str]] = {}
        self._is_internal_cache: Dict[str, bool] = {}
        self._cell_tags_cache: Dict[str, Set[str]] = {}

    def plan_path(
        self,
        prompt: str,
        start_sig: Optional[Union[AlgebraicSignature, PortSignature]] = None,
        goal_sig: Optional[Union[AlgebraicSignature, PortSignature]] = None,
        start_type: Optional[str] = None,
        start_state: Optional[str] = None,
        goal_type: Optional[str] = None,
        goal_state: Optional[str] = None,
        beam_width: int = 5,
        max_steps: int = 12,
        return_tuple: Optional[bool] = None
    ) -> Union[List[Cell], Tuple[List[Cell], Set[str]]]:
        is_tuple_requested = return_tuple if return_tuple is not None else (start_sig is None and goal_sig is None)

        astar = SemanticStateAStar(self.orchestrator, self.rag)
        frame = SemanticFrame.build(prompt, astar_instance=astar)
        required_intents = astar.extract_required_intents(prompt)
        intents_set = set(required_intents)
        has_explicit_sink = frame.has_explicit_sink
        clauses = list(frame.clauses)

        if start_sig is not None:
            if hasattr(start_sig, "signature") and hasattr(start_sig, "name"):
                start_port = start_sig
            elif hasattr(start_sig, "type_name") and hasattr(start_sig, "state"):
                start_port = PortSignature("input_data", start_sig)
            else:
                start_port = PortSignature("input_data", start_sig)
        elif start_type is not None or start_state is not None:
            start_port = PortSignature(
                "input_data",
                AlgebraicSignature(start_type or "str", start_state or "source_identifier")
            )
        else:
            has_file_ingress = bool(
                frame.path_literals
                or frame.format_tokens
                or any(re.search(rf"\b{re.escape(v)}\b", prompt.lower()) for v in astar.STAGE_ROLE_TAGS[1])
            )
            if has_file_ingress:
                start_port = PortSignature(
                    "input_data",
                    AlgebraicSignature("str", "source_identifier")
                )
            else:
                start_port = None

        if goal_sig is not None:
            if hasattr(goal_sig, "signature") and hasattr(goal_sig, "name"):
                goal_port = goal_sig
            elif hasattr(goal_sig, "type_name") and hasattr(goal_sig, "state"):
                goal_port = PortSignature("output_data", goal_sig)
            else:
                goal_port = PortSignature("output_data", goal_sig)
        elif goal_type is not None or goal_state is not None:
            goal_port = PortSignature(
                "output_data",
                AlgebraicSignature(goal_type or "str", goal_state or "filepath_written")
            )
        else:
            goal_port = None

        cell_clause_map: Dict[str, Set[int]] = {}

        if len(self.orchestrator.loaded_cells) <= 100:
            candidate_pool = list(self.orchestrator.loaded_cells.values())
            for c in candidate_pool:
                cell_clause_map.setdefault(c.cell_id, set()).update(range(len(clauses)))
        else:
            # 1. Verified core seeds
            candidate_pool = [
                c for c in self.orchestrator.loaded_cells.values()
                if getattr(c, "verified", False) and not any(h in c.cell_id for h in ["_DEFAULT", "_INTERNAL", "_GROUP_", "_TYPING"])
            ]

            # 2. Context from RAG if available, otherwise intent keyword matching
            if self.rag is not None:
                try:
                    # Global prompt context
                    context = self.rag.get_relevant_context(prompt, top_k=40)
                    for entry in context:
                        if isinstance(entry, dict):
                            cid = entry.get("cell_id", "")
                            c = self.orchestrator.loaded_cells.get(cid)
                            if c:
                                candidate_pool.append(c)

                    # Multi-clause sub-intent retrieval for composite pipelines
                    if len(clauses) > 1:
                        for clause_idx, clause in enumerate(clauses):
                            sub_context = self.rag.get_relevant_context(clause, top_k=15)
                            for entry in sub_context:
                                if isinstance(entry, dict):
                                    cid = entry.get("cell_id", "")
                                    c = self.orchestrator.loaded_cells.get(cid)
                                    if c:
                                        candidate_pool.append(c)
                                        cell_clause_map.setdefault(c.cell_id, set()).add(clause_idx)
                except Exception as e:
                    logger.warning(f"[ROUTER] RAG candidate retrieval error: {e}")
            else:
                if hasattr(self.orchestrator, "_cells_by_keyword") and self.orchestrator._cells_by_keyword:
                    matched_counts: Dict[Cell, int] = {}
                    for clause_idx, c_intents in enumerate(frame.clause_intents):
                        for intent in c_intents:
                            for c in self.orchestrator._cells_by_keyword.get(intent.lower(), []):
                                if getattr(c, "verified", False):
                                    continue
                                is_int = self._is_internal_cache.get(c.cell_id)
                                if is_int is None:
                                    cid_l = c.cell_id.lower()
                                    is_int = any(h in cid_l for h in ("_default", "_internal", "_group_", "typing_"))
                                    self._is_internal_cache[c.cell_id] = is_int
                                if is_int:
                                    continue
                                matched_counts[c] = matched_counts.get(c, 0) + 1
                                cell_clause_map.setdefault(c.cell_id, set()).add(clause_idx)
                    other_scored = [c for c, count in matched_counts.items() if count >= 2]
                    candidate_pool.extend(other_scored[:40])
                else:
                    other_scored = []
                    for c in self.orchestrator.loaded_cells.values():
                        if getattr(c, "verified", False):
                            continue
                        is_int = self._is_internal_cache.get(c.cell_id)
                        if is_int is None:
                            cid_l = c.cell_id.lower()
                            is_int = any(h in cid_l for h in ("_default", "_internal", "_group_", "typing_"))
                            self._is_internal_cache[c.cell_id] = is_int
                        if is_int:
                            continue
                        overlap = len(intents_set & c.keywords)
                        if overlap >= 2:
                            other_scored.append((c, overlap))
                    other_scored.sort(key=lambda x: x[1], reverse=True)
                    candidate_pool.extend([c for c, _ in other_scored[:30]])

            # Dynamic semantic attribution: map candidates to clauses based on semantic intent overlap
            for c in candidate_pool:
                c_tags = self._cell_tags_cache.get(c.cell_id)
                if c_tags is None:
                    c_tags = (
                        set(getattr(c, "semantic_tags", []))
                        | set(getattr(c, "keywords", []))
                        | {c.cell_id.lower(), getattr(c, "domain_name", "").lower()}
                        | {p.type_name.lower() for p in list(c.inputs.values()) + list(c.outputs.values()) if hasattr(p, "type_name")}
                    )
                    self._cell_tags_cache[c.cell_id] = c_tags
                is_sink = c.stage == 3 or getattr(c, "primary_output", None) and c.primary_output.state in ("filepath_written", "saved", "exported")
                is_source = c.stage == 1

                matched_clause = False
                for clause_idx, c_intents in enumerate(frame.clause_intents):
                    # Stage-role invariants:
                    # Sink cells only belong to clauses with sink verbs
                    if is_sink and not any(v in c_intents for v in astar.STAGE_ROLE_TAGS[3]):
                        continue
                    # Source cells only belong to clauses with source verbs
                    if is_source and not any(v in c_intents for v in astar.STAGE_ROLE_TAGS[1]):
                        continue

                    # Filter out generic file/format tokens when matching non-I/O clauses
                    meaningful_intents = {ci for ci in c_intents if ci not in ("csv", "json", "file", "data", "dataset", "table")}
                    if not meaningful_intents:
                        meaningful_intents = c_intents

                    if any(ci in tag or tag in ci for ci in meaningful_intents for tag in c_tags if len(ci) >= 3):
                        cell_clause_map.setdefault(c.cell_id, set()).add(clause_idx)
                        matched_clause = True

                if not matched_clause and c.cell_id not in cell_clause_map:
                    if is_source and not any(0 in s for s in cell_clause_map.values()):
                        cell_clause_map[c.cell_id] = {0}
                    elif is_sink and not any((len(clauses) - 1) in s for s in cell_clause_map.values()):
                        cell_clause_map[c.cell_id] = {len(clauses) - 1}

            candidate_pool = list({c.cell_id: c for c in candidate_pool}.values())

        # Dynamic Terminal Goal Condition: prune unsolicited sink/export nodes
        def _is_sink_cell(c: Cell) -> bool:
            if c.stage == 3:
                return True
            out_sig = getattr(c, "primary_output", None)
            if out_sig:
                if out_sig.state in ("filepath_written", "saved", "exported"):
                    return True
                if out_sig.type_name in ("None", "NoneType", "Unit"):
                    return True
            return False

        if not has_explicit_sink:
            candidate_pool = [c for c in candidate_pool if not _is_sink_cell(c)]

        clause_intents_list = [list(ci) for ci in frame.clause_intents]

        resolved_path = astar.search(
            start_port,
            goal_port,
            required_intents,
            candidate_pool=candidate_pool,
            clauses=clauses,
            cell_clause_map=cell_clause_map,
            has_explicit_sink=has_explicit_sink,
            clause_intents_list=clause_intents_list,
            frame=frame
        )

        if resolved_path:
            return (resolved_path, set()) if is_tuple_requested else resolved_path

        fallback_start_sig = start_port.signature if start_port else AlgebraicSignature("str", "source_identifier")
        return self._beam_search_fallback(prompt, fallback_start_sig, goal_port.signature if goal_port else None, beam_width, max_steps, is_tuple_requested)

    def _beam_search_fallback(
        self,
        prompt: str,
        start_sig: AlgebraicSignature,
        target_goal_sig: Optional[AlgebraicSignature],
        beam_width: int,
        max_steps: int,
        is_tuple_requested: bool
    ) -> Union[List[Cell], Tuple[List[Cell], Set[str]]]:
        prompt_keywords = set(re.findall(r"[a-zA-Z_]+", prompt.lower()))
        current_sig = start_sig
        beam: List[Tuple[List[str], AlgebraicSignature, float]] = [([], current_sig, 0.0)]
        visited_sequences: Set[str] = set()

        for step in range(max_steps):
            candidates: List[Tuple[List[str], AlgebraicSignature, float]] = []

            for path, sig, score in beam:
                path_cells = self._ids_to_cells(path)
                if is_goal_satisfied(sig, target_goal_sig, path_cells) and path:
                    return (path_cells, set()) if is_tuple_requested else path_cells

                seq_key = "->".join(path)
                if seq_key in visited_sequences:
                    continue
                visited_sequences.add(seq_key)

                try:
                    next_nodes = self._score_candidates(sig, prompt, prompt_keywords, top_k=25)
                except Exception as e:
                    logger.error(f"[ROUTER] Scoring failed at step {step}: {e}")
                    next_nodes = []

                if not next_nodes:
                    next_nodes = self._keyword_fallback(sig, prompt_keywords, top_k=10)

                for cid, node_score in next_nodes:
                    if cid in path:
                        continue

                    cell = self.orchestrator.loaded_cells.get(cid)
                    if not cell:
                        continue

                    new_path = path + [cid]
                    new_sig = cell.primary_output.signature if hasattr(cell.primary_output, "signature") else cell.primary_output
                    new_score = score + node_score - (len(new_path) * 0.01)
                    candidates.append((new_path, new_sig, new_score))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[2], reverse=True)
            beam = candidates[:beam_width]

        if beam:
            best_path, best_sig, _ = max(beam, key=lambda x: x[2])
            if best_path:
                res_cells = self._ids_to_cells(best_path)
                return (res_cells, set()) if is_tuple_requested else res_cells

        return ([], set()) if is_tuple_requested else []

    def _ids_to_cells(self, ids: List[str]) -> List[Cell]:
        cells: List[Cell] = []
        for cid in ids:
            cell = self.orchestrator.loaded_cells.get(cid)
            if cell:
                cells.append(cell)
        return cells

    def _score_candidates(
        self,
        current_sig: AlgebraicSignature,
        prompt: str,
        prompt_keywords: Set[str],
        top_k: int = 25
    ) -> List[Tuple[str, float]]:
        """Scores candidate cells with strict type pre-filtering and domain alignment."""
        if self.rag is None:
            return []

        try:
            raw_candidates = self.rag.get_relevant_context(prompt, top_k=top_k * 2)
        except Exception as e:
            logger.error(f"[ROUTER] RAG query failed: {e}")
            return []

        if not raw_candidates:
            return []

        # Handle list of dicts from RAG directly
        entries: List[Dict[str, Any]] = []
        if isinstance(raw_candidates, list):
            for item in raw_candidates:
                if isinstance(item, dict):
                    entries.append(item)
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    entries.append({
                        "cell_id": str(item[0]),
                        "score": float(item[1]) if len(item) > 1 else 0.5
                    })

        scored: List[Tuple[str, float]] = []
        prompt_lower = prompt.lower()

        # Infer active domain dynamically
        domain_hint = None
        if entries:
            top_dom = entries[0].get("domain", "")
            if top_dom and top_dom not in ("generic", "python_core"):
                domain_hint = top_dom.lower()

        if not domain_hint and entries:
            domain_counts: Dict[str, float] = {}
            for entry in entries[:5]:
                d = (entry.get("domain") or "").lower()
                if d and d not in ("generic", "python_core", "macro"):
                    domain_counts[d] = domain_counts.get(d, 0.0) + float(entry.get("score", 0.5))
            if domain_counts:
                domain_hint = max(domain_counts, key=domain_counts.get)

        for entry in entries:
            cid = entry.get("cell_id", "")
            if not cid:
                continue

            cell = self.orchestrator.loaded_cells.get(cid)
            if not cell:
                continue

            # STRICT TYPE PRE-FILTERING (C3 FIX):
            # If the candidate cell cannot accept the current signature, reject immediately!
            if not current_sig.unifies_with(cell.primary_input):
                continue

            base_score = float(entry.get("score", 0.5))

            if domain_hint:
                cell_domain = (cell.domain_name or "").lower()
                if domain_hint == cell_domain:
                    base_score *= 1.4
                elif cell_domain not in ("generic", "macro", "python_core"):
                    base_score *= 0.3

            cell_kws = set(k.lower() for k in cell.keywords)
            overlap = len(prompt_keywords & cell_kws)
            base_score += overlap * 0.25

            # Filter internal/infrastructure nodes via cell metadata instead of string matching
            if getattr(cell, 'stage', 2) not in (1, 2, 3):
                base_score *= 0.3

            scored.append((cid, base_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _keyword_fallback(
        self,
        current_sig: AlgebraicSignature,
        prompt_keywords: Set[str],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Pure keyword matching fallback when RAG returns nothing or crashes."""
        results: List[Tuple[str, float]] = []
        for cid, cell in self.orchestrator.loaded_cells.items():
            if not current_sig.unifies_with(cell.primary_input):
                continue
            cell_kws = set(k.lower() for k in cell.keywords)
            overlap = len(prompt_keywords & cell_kws)
            if overlap > 0:
                results.append((cid, overlap * 0.5))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class FastPathRouter:
    def __init__(self, orchestrator: LatticeOrchestrator, rag: Any):
        self.orchestrator = orchestrator
        self.rag = rag

    def try_fast_path(self, prompt: str, threshold: float = 0.92) -> Optional[List[str]]:
        result = self.rag.find_closest_cell_by_embedding(prompt)
        if not result or result.get("score", 0.0) < threshold:
            return None

        cid = result.get("cell_id", "")
        cell = self.orchestrator.loaded_cells.get(cid)
        if isinstance(cell, MacroCell):
            return cell.sub_cells
        return None
