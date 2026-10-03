"""
src/fixtures.py - Neuro-Symbolic Topological Lattice (NSTL)
Dynamic fixture synthesis for sandboxed execution and verification.

Zero hardcoded datasets or column names:
- Derives required input files from file-asset literals bound to path ports on the pipeline.
- Infers column schemas from identifier literals and column-role bindings across planned cells.
- Satisfies declared cell preconditions via 'fixture_needs' metadata declared on lattice nodes (e.g., 'missing_values').
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class FixtureSynthesizer:
    """
    Synthesizes input files and test fixtures in the sandbox working directory
    based on the declared requirements of the planned pipeline cells.
    """

    @classmethod
    def synthesize_for_pipeline(
        cls,
        pipeline_bindings: List[Tuple[Any, Dict[str, Any]]],
        working_dir: str | Path,
        extracted_literals: Optional[List[Tuple[int, str, Any]]] = None,
        num_rows: int = 50,
    ) -> List[Path]:
        """
        Inspects pipeline_bindings for source file inputs and generates matching fixture files.
        Returns a list of created file paths.
        """
        import numpy as np
        import pandas as pd

        target_dir = Path(working_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        created_files: List[Path] = []

        if not pipeline_bindings:
            return created_files

        # 1. Aggregate fixture_needs from all cells in the pipeline
        fixture_needs: Set[str] = set()
        for cell, _ in pipeline_bindings:
            needs = getattr(cell, "fixture_needs", None)
            if needs and isinstance(needs, (list, tuple, set)):
                fixture_needs.update(str(n).lower() for n in needs)

        # 2. Extract column identifiers bound across the pipeline or present in literals
        candidate_cols: List[str] = []
        seen_cols: Set[str] = set()

        if extracted_literals:
            for _, kind, val in extracted_literals:
                sval = str(val).strip("'\"")
                if kind in ("identifier", "quoted_str"):
                    # Exclude file names and extensions
                    if "." not in sval and sval.isidentifier() and sval not in seen_cols:
                        seen_cols.add(sval)
                        candidate_cols.append(sval)

        for cell, bindings in pipeline_bindings:
            if not isinstance(bindings, dict):
                continue
            for p_name, b_val in bindings.items():
                if isinstance(b_val, str):
                    clean = b_val.strip("'\"")
                    if clean.isidentifier() and "." not in clean and clean not in seen_cols:
                        port_sig = getattr(cell, "inputs", {}).get(p_name)
                        role = getattr(port_sig, "port_role", "") if port_sig else ""
                        if role in ("literal_parameter", "target_input", "feature_input") or "col" in p_name.lower():
                            seen_cols.add(clean)
                            candidate_cols.append(clean)
                elif isinstance(b_val, (list, tuple)):
                    for item in b_val:
                        if isinstance(item, str):
                            clean = item.strip("'\"")
                            if clean.isidentifier() and "." not in clean and clean not in seen_cols:
                                seen_cols.add(clean)
                                candidate_cols.append(clean)

        if not candidate_cols:
            candidate_cols = ["X", "Y"]

        # 3. Detect file inputs from source cells
        for cell, bindings in pipeline_bindings:
            if not isinstance(bindings, dict):
                continue
            for p_name, b_val in bindings.items():
                if not b_val or not isinstance(b_val, str):
                    continue
                clean_path = b_val.strip("'\"")
                _, ext = os.path.splitext(clean_path)
                if not ext:
                    continue

                dest_file = target_dir / os.path.basename(clean_path)
                if dest_file.exists():
                    continue

                ext_lower = ext.lower()
                if ext_lower in (".csv", ".tsv", ".txt", ".parquet"):
                    # Synthesize tabular data
                    data: Dict[str, Any] = {}
                    np.random.seed(42)
                    for col in candidate_cols:
                        # Generate smooth continuous numeric data
                        data[col] = np.random.uniform(10.0, 100.0, size=num_rows)

                    # Inject missing values if required by cell metadata
                    if "missing_values" in fixture_needs or "nulls" in fixture_needs:
                        for col in candidate_cols:
                            nan_indices = np.random.choice(num_rows, size=max(1, num_rows // 10), replace=False)
                            data[col][nan_indices] = np.nan

                    df = pd.DataFrame(data)

                    if ext_lower == ".csv":
                        df.to_csv(dest_file, index=False)
                    elif ext_lower in (".tsv", ".txt"):
                        df.to_csv(dest_file, sep="\t", index=False)
                    elif ext_lower == ".parquet":
                        df.to_parquet(dest_file, index=False)

                    created_files.append(dest_file)

        return created_files
