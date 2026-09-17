# STATUS: partially wired / candidate for future removal. Do not expand until usage is confirmed.
"""
src/macro_harvester.py - Neuro-Symbolic Topological Lattice (NSTL)
Dynamic Macro Harvesting & Hierarchical Abstraction (Phase 4 / T4.3).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any

from log_config import get_logger

try:
    from .lattice import LatticeOrchestrator, Cell, MicroCell, MacroCell, PortSignature, AlgebraicSignature
    from .unification import UnificationGate, ExecutionContext
    from .tokenizer import CellTokenizer
except (ImportError, ValueError):
    from lattice import LatticeOrchestrator, Cell, MicroCell, MacroCell, PortSignature, AlgebraicSignature
    from unification import UnificationGate, ExecutionContext
    from tokenizer import CellTokenizer

logger = get_logger("macro_harvester")


class MacroHarvester:
    """
    Synthesizes and registers MacroCell abstractions from verified subgraphs or cell sequences.
    """

    @classmethod
    def harvest_macro(
        cls,
        cell_ids: List[str],
        orchestrator: LatticeOrchestrator,
        macro_id: Optional[str] = None,
        domain_name: Optional[str] = None,
        docstring: Optional[str] = None
    ) -> MacroCell:
        """
        Harvests a composite MacroCell from a list of cell IDs.
        Verifies internal type-monadic consistency, constructs the outer port signature,
        and registers the new MacroCell in the orchestrator.
        """
        if not cell_ids:
            raise ValueError("[MACRO_HARVESTER] Cannot create macro from empty cell list.")

        loaded = orchestrator.loaded_cells
        cells: List[Cell] = []
        for cid in cell_ids:
            c = loaded.get(cid)
            if not c:
                raise ValueError(f"[MACRO_HARVESTER] Cell '{cid}' not found in orchestrator.")
            cells.append(c)

        # 1. Verify internal compositional validity
        try:
            from .planner import LatticePlanner
            from .unification import Substitution
        except (ImportError, ValueError):
            from planner import LatticePlanner
            from unification import Substitution

        planner = LatticePlanner(orchestrator=orchestrator)
        chain: List[Cell] = [cells[0]]
        accum_sigma = Substitution()
        for c in cells[1:]:
            step_sigma = planner._verify_transition(chain, c, accum_sigma)
            if step_sigma is None:
                raise ValueError(
                    f"[MACRO_HARVESTER] Cell '{chain[-1].cell_id}' does not form a type-valid composition with '{c.cell_id}'."
                )
            accum_sigma = step_sigma
            chain.append(c)

        # 2. Derive Outer Interface (Inputs & Outputs)
        first_cell = cells[0]
        last_cell = cells[-1]

        # Stage resolution: preserve source (1) or sink (3) semantics, otherwise intermediate (2)
        resolved_stage = first_cell.stage if first_cell.stage == 1 else (last_cell.stage if last_cell.stage == 3 else 2)

        # Composite inputs: all inputs of the first cell
        composite_inputs: Dict[str, PortSignature] = dict(first_cell.inputs)

        # For intermediate cells, collect unfulfilled required inputs (e.g. hyper-parameters)
        preceding_cells: List[Cell] = []
        for c in cells[1:]:
            for p_name, p_sig in c.inputs.items():
                # If this port matches none of the previous cells' outputs, it's external
                is_internal = any(
                    out_sig.signature.matches(p_sig.signature)
                    for prev_c in preceding_cells
                    for out_sig in prev_c.outputs.values()
                )
                if not is_internal and p_name not in composite_inputs:
                    composite_inputs[p_name] = p_sig
            preceding_cells.append(c)

        # Composite outputs: all outputs of the last cell
        composite_outputs: Dict[str, PortSignature] = dict(last_cell.outputs)

        # Aggregate tokens and keywords
        combined_keywords: List[str] = []
        combined_deps: List[str] = []
        for c in cells:
            for kw in getattr(c, "keywords", []):
                if kw not in combined_keywords:
                    combined_keywords.append(kw)
            for dep in getattr(c, "dependencies", []):
                if dep not in combined_deps:
                    combined_deps.append(dep)

        resolved_macro_id = macro_id or f"MACRO_{first_cell.cell_id}_{last_cell.cell_id}"
        resolved_domain = domain_name or first_cell.domain_name
        internal_topology = {cells[i].cell_id: [cells[i+1].cell_id] for i in range(len(cells)-1)}

        macro_cell = MacroCell(
            cell_id=resolved_macro_id,
            stage=resolved_stage,
            keywords=combined_keywords,
            inputs=composite_inputs,
            outputs=composite_outputs,
            domain_name=resolved_domain,
            dependencies=combined_deps,
            code_template="\n".join(c.code_template for c in cells if getattr(c, "code_template", "")),
            sub_cells=cell_ids,
            algorithmic_steps=[getattr(c, "docstring", c.cell_id) for c in cells],
            internal_topology=internal_topology,
            docstring=docstring or f"Composite macro morphism: {' -> '.join(cell_ids)}",
            verified=True
        )

        # Cache resolved sub-cells on macro instance
        macro_cell._resolved_sub_cells = {c.cell_id: c for c in cells}

        # Register in orchestrator
        orchestrator.loaded_cells[macro_cell.cell_id] = macro_cell

        # Rebuild topology so the macro is immediately routable: build_topology
        # refreshes the token index (used by the router's IDF retrieval), the
        # adjacency maps, and the bridge-cell registry. Without this, a freshly
        # harvested macro is invisible to LatticeRouter until the next restart.
        try:
            orchestrator.build_topology()
        except Exception as exc:
            logger.warning(f"[MACRO_HARVESTER] Topology rebuild after harvest failed: {exc}")

        logger.info(f"[MACRO_HARVESTER] Successfully harvested and registered MacroCell '{resolved_macro_id}'")

        return macro_cell
