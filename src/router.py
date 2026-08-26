"""
src/router.py - Neuro-Symbolic Topological Lattice (NSTL)
Implements Semantic Tunneling, Type-Monadic Beam Search Pathfinding,
and Goal-Directed MCTS Typestate Bridging.
"""

from __future__ import annotations
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple, Any

import numpy as np
import torch
from log_config import get_logger
from lattice import LatticeOrchestrator, Cell, MicroCell, MacroCell, AlgebraicSignature

logger = get_logger('router')


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
        return (self.q_value / self.visits) + c_param * math.sqrt(math.log(parent_visits) / self.visits)


class MCTSEngine:
    def __init__(self, orchestrator: LatticeOrchestrator):
        self.orchestrator = orchestrator

    def search(self, start_sig: AlgebraicSignature, target_sig: AlgebraicSignature, max_depth: int = 6, timeout_sec: float = 1.5) -> List[Cell]:
        root = MCTSNode(cell_id="ROOT", signature=start_sig)
        start_time = time.time()

        for _ in range(100):
            if time.time() - start_time > timeout_sec:
                break

            leaf = self._select(root)

            if not leaf.signature.unifies_with(target_sig) and leaf.visits > 0:
                self._expand(leaf)
                if leaf.children:
                    leaf = random.choice(leaf.children)

            reward = self._simulate(leaf, target_sig, max_depth)
            self._backpropagate(leaf, reward)

        return self._extract_best_path(root, target_sig)

    def _select(self, node: MCTSNode) -> MCTSNode:
        curr = node
        while curr.children:
            unexplored = [c for c in curr.children if c.visits == 0]
            if unexplored:
                return unexplored[0]
            curr = max(curr.children, key=lambda c: c.ucb1())
        return curr

    def _expand(self, node: MCTSNode):
        if node.cell_id == "ROOT":
            candidates = [c for c in self.orchestrator.loaded_cells.values() if isinstance(c, MicroCell)]
        else:
            candidates = self.orchestrator.get_neighbors(node.cell_id)

        for cell in candidates:
            if isinstance(cell, MicroCell) and node.signature.unifies_with(cell.primary_input):
                child = MCTSNode(cell_id=cell.cell_id, signature=cell.primary_output, parent=node)
                node.children.append(child)

    def _simulate(self, node: MCTSNode, target_sig: AlgebraicSignature, max_depth: int) -> float:
        curr_sig = node.signature
        depth = 0

        while not curr_sig.unifies_with(target_sig) and depth < max_depth:
            neighbors = [
                c for c in self.orchestrator.loaded_cells.values()
                if isinstance(c, MicroCell) and curr_sig.unifies_with(c.primary_input)
            ]
            if not neighbors:
                return 0.0
            next_cell = random.choice(neighbors)
            curr_sig = next_cell.primary_output
            depth += 1

        return 1.0 if curr_sig.unifies_with(target_sig) else 0.0

    def _backpropagate(self, node: MCTSNode, reward: float):
        curr: Optional[MCTSNode] = node
        while curr is not None:
            curr.visits += 1
            curr.q_value += reward
            curr = curr.parent

    def _extract_best_path(self, root: MCTSNode, target_sig: AlgebraicSignature) -> List[Cell]:
        path = []
        curr = root
        while curr.children:
            best_child = max(curr.children, key=lambda c: c.visits)
            cell = self.orchestrator.loaded_cells.get(best_child.cell_id)
            if not cell:
                break
            path.append(cell)
            if best_child.signature.unifies_with(target_sig):
                return path
            curr = best_child
        return []


@dataclass
class BeamCandidate:
    path: List[Cell]
    current_signature: AlgebraicSignature
    score: float = 0.0
    virtual_edges: Set[str] = field(default_factory=set)


class LatticeRouter:
    def __init__(self, orchestrator: LatticeOrchestrator, rag_engine=None):
        self.orchestrator = orchestrator
        self.rag_engine = rag_engine
        self.mcts = MCTSEngine(orchestrator)

    def _split_intent_into_goals(self, intent: str) -> List[str]:
        pattern = r'(?:\s*;\s*|\s*,\s*|\s*\.\s+|\n+)'
        raw_goals = re.split(pattern, intent, flags=re.IGNORECASE)
        goals = [g.strip() for g in raw_goals if g.strip()]
        return goals if goals else [intent.strip()]

    def plan_path(
        self,
        user_intent: str,
        initial_type: str = "str",
        initial_state: str = "source_identifier",
        beam_width: int = 3
    ) -> Tuple[List[Cell], Set[str]]:
        goals = self._split_intent_into_goals(user_intent)
        start_sig = AlgebraicSignature(initial_type, initial_state)

        beam: List[BeamCandidate] = [
            BeamCandidate(path=[], current_signature=start_sig, score=0.0)
        ]

        from inference import ModelManager
        model_mgr = ModelManager.get_instance()

        for goal in goals:
            goal_embeddings = model_mgr.get_embeddings([goal])
            if not goal_embeddings:
                continue
            goal_vec = np.array(goal_embeddings, dtype=np.float32)
            goal_vec = goal_vec / (np.linalg.norm(goal_vec, axis=1, keepdims=True) + 1e-9)

            next_beam: List[BeamCandidate] = []

            for state in beam:
                candidates = self._score_candidates(goal_vec, state.current_signature)

                for score, cell in candidates[:beam_width * 2]:
                    last_stage = state.path[-1].stage if state.path else 0
                    cell_stage = cell.stage or 2

                    # Penalize stage regression (e.g., sink -> source)
                    if last_stage > 0 and cell_stage < last_stage:
                        score *= 0.6

                    if state.current_signature.unifies_with(cell.primary_input):
                        new_path = state.path + [cell]
                        new_sig = cell.primary_output
                        new_edges = set(state.virtual_edges)

                        if state.path:
                            last_id = state.path[-1].cell_id
                            neighbors = [n.cell_id for n in self.orchestrator.get_neighbors(last_id)]
                            if cell.cell_id not in neighbors:
                                new_edges.add(cell.cell_id)

                        next_beam.append(
                            BeamCandidate(
                                path=new_path,
                                current_signature=new_sig,
                                score=state.score + math.log(max(score, 1e-4)),
                                virtual_edges=new_edges
                            )
                        )
                    else:
                        bridge = self.mcts.search(state.current_signature, cell.primary_input)
                        if bridge:
                            new_path = state.path + bridge + [cell]
                            new_sig = cell.primary_output
                            new_edges = set(state.virtual_edges)
                            for b in bridge:
                                new_edges.add(b.cell_id)
                            new_edges.add(cell.cell_id)

                            next_beam.append(
                                BeamCandidate(
                                    path=new_path,
                                    current_signature=new_sig,
                                    score=state.score + math.log(max(score * 0.8, 1e-4)),
                                    virtual_edges=new_edges
                                )
                            )

            if next_beam:
                next_beam.sort(key=lambda b: b.score, reverse=True)
                unique_beam = []
                seen_paths = set()
                for b in next_beam:
                    key = tuple(c.cell_id for c in b.path)
                    if key not in seen_paths:
                        seen_paths.add(key)
                        unique_beam.append(b)
                beam = unique_beam[:beam_width]
            else:
                logger.warning(f"[ROUTER HALT] No type-valid transitions found for sub-goal: '{goal}'")
                break

        if not beam or not beam[0].path:
            return [], set()

        best = max(beam, key=lambda b: b.score)
        logger.info(f"[ROUTER COMPLETE] Path: {[c.cell_id for c in best.path]}")
        return best.path, best.virtual_edges

    def _score_candidates(
        self,
        goal_vec: np.ndarray,
        current_sig: AlgebraicSignature,
        top_k: int = 50
    ) -> List[Tuple[float, Cell]]:
        if self.rag_engine is None or self.rag_engine.index is None:
            return []

        k = min(top_k, self.rag_engine.index.ntotal)
        distances, indices = self.rag_engine.index.search(goal_vec, k=k)

        scored: List[Tuple[float, Cell]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx not in self.rag_engine.id_to_schema:
                continue

            cell_id = self.rag_engine.id_to_schema[idx].get("cell_id")
            cell = self.orchestrator.loaded_cells.get(cell_id)
            if not cell:
                continue

            # Never route through macros — they are expanded by the planner
            if cell.node_type == "macro":
                continue

            # Skip cells with empty templates (non-executable)
            if not cell.code_template or not cell.code_template.strip():
                continue

            sim = float(dist)

            if current_sig.unifies_with(cell.primary_input):
                type_multiplier = 1.0
            else:
                type_multiplier = 0.5

            scored.append((sim * type_multiplier, cell))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored
