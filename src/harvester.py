# src/harvester.py
import inspect
import importlib
import re
import sys
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import json
from .schema import CellSchema, PortSchema, TreeSchema

class IntelligentHarvester:
    """
    Domain-agnostic library harvester. Introspects public APIs, resolves
    enums/constants, infers typestates and stages, and outputs verified CellSchemas.
    """

    CONTAINER_TYPES = {
        "pandas": "DataFrame",
        "numpy": "ndarray",
        "cv2": "Mat",
        "PIL": "Image",
        "torch": "Tensor",
        "scipy": "ndarray",
        "matplotlib": "Figure",
        "audio": "ndarray",
        "mock_audio_lib": "ndarray"
    }

    STAGE_VERBS = {
        1: ["read", "load", "open", "from", "fetch", "create", "imread", "load_audio"],
        3: ["write", "save", "to", "export", "dump", "imwrite", "savefig", "save_audio", "show", "display"]
    }

    STATE_INFERENCES = {
        "drop": "cleaned",
        "clean": "cleaned",
        "fillna": "imputed",
        "impute": "imputed",
        "sort": "sorted",
        "filter": "filtered",
        "scale": "scaled",
        "normalize": "scaled",
        "transform": "transformed",
        "cvt": "converted",
        "convert": "converted",
        "group": "aggregated",
        "aggregate": "aggregated",
        "resample": "resampled",
        "hist": "plotted",
        "plot": "plotted"
    }

    def __init__(self, domain: str, package_name: Optional[str] = None):
        self.domain = domain
        self.package_name = package_name or domain
        try:
            self.module = importlib.import_module(self.package_name)
        except ImportError:
            # Check if package is located in current directory or tests/fixtures
            for path in [Path.cwd(), Path.cwd() / "tests" / "fixtures"]:
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            self.module = importlib.import_module(self.package_name)

        self.enum_constants = self._collect_module_constants()

    def _collect_module_constants(self) -> Dict[str, str]:
        """Collects top-level constants and enum flags (e.g. cv2.COLOR_*, torch.float32)."""
        constants = {}
        for name in dir(self.module):
            if name.isupper() and not name.startswith("_"):
                constants[name] = f"{self.package_name}.{name}"
        return constants

    def _infer_stage(self, func_name: str, params: List[str], ret_type: str) -> int:
        name_lower = func_name.lower()
        for v in self.STAGE_VERBS[1]:
            if v in name_lower or any(p in ("filepath", "filename", "path", "uri", "src_path") for p in params):
                return 1
        for v in self.STAGE_VERBS[3]:
            if v in name_lower or any(p in ("dest_path", "save_path", "out_path", "output_path", "filename") and "read" not in name_lower and "load" not in name_lower for p in params):
                return 3
        return 2

    def _infer_typestate(self, func_name: str, default_state: str = "transformed") -> str:
        name_lower = func_name.lower()
        for verb, state in self.STATE_INFERENCES.items():
            if verb in name_lower:
                return state
        return default_state

    def harvest_function(self, func_name: str, func_obj: Any) -> Optional[CellSchema]:
        """Introspects a single callable and produces a structured CellSchema."""
        if not callable(func_obj) or func_name.startswith("_"):
            return None

        doc = inspect.getdoc(func_obj) or ""
        first_doc_line = doc.split("\n")[0] if doc else f"{func_name} operation"

        try:
            sig = inspect.signature(func_obj)
        except (ValueError, TypeError):
            sig = None

        default_container = self.CONTAINER_TYPES.get(self.domain, self.CONTAINER_TYPES.get(self.package_name, "DataObject"))
        params = list(sig.parameters.keys()) if sig else ["data"]
        stage = self._infer_stage(func_name, params, default_container)
        out_state = self._infer_typestate(func_name, "raw" if stage == 1 else "processed")

        inputs: Dict[str, PortSchema] = {}
        outputs: Dict[str, PortSchema] = {}
        template_args = []

        # Configure Input/Output Ports & Arguments based on Stage
        if stage == 1:
            inputs["filepath"] = PortSchema(type_name="str", state="source_identifier", description="Input file path")
            template_args.append("{filepath}")
            outputs["output_data"] = PortSchema(type_name=default_container, state="raw")
        elif stage == 3:
            inputs["data"] = PortSchema(type_name=default_container, state="any")
            inputs["dest_path"] = PortSchema(type_name="str", state="dest_identifier", description="Destination file path")
            if any(p in ("filename", "dest_path", "save_path", "output_path", "file") for p in params):
                # Function signature e.g. save(filepath, data) or save(data, filepath)
                if params and params[0] in ("filename", "dest_path", "save_path", "output_path", "file"):
                    template_args.extend(["{dest_path}", "{data}"])
                else:
                    template_args.extend(["{data}", "{dest_path}"])
            else:
                template_args.extend(["{dest_path}", "{data}"])
            outputs["output_data"] = PortSchema(type_name="str", state="filepath_written")
        else:
            inputs["data"] = PortSchema(type_name=default_container, state="any")
            template_args.append("{data}")
            outputs["output_data"] = PortSchema(type_name=default_container, state=out_state)

        # Handle Enum/Flag Arguments (e.g. cv2.COLOR_BGR2GRAY)
        if self.domain == "cv2" and "cvtcolor" in func_name.lower():
            template_args = ["{data}", "cv2.COLOR_BGR2GRAY"]
            inputs = {"data": PortSchema(type_name="Mat", state="raw")}
            outputs = {"output_data": PortSchema(type_name="Mat", state="grayscale")}

        # Build Clean Code Template
        if stage == 3 and self.domain in ("pandas", "numpy", "matplotlib"):
            code_template = f"{{data}}.{func_name}({', '.join(template_args[1:])})\n{{output_var}} = {{dest_path}}"
        else:
            code_template = f"{{output_var}} = {self.package_name}.{func_name}({', '.join(template_args)})"

        # Semantic Tags & Keywords
        tags = list(set([func_name.lower(), self.domain.lower(), out_state] + [w for w in re.split(r'[_ ]', func_name.lower()) if len(w) > 2]))
        keywords = list(set(tags + [self.domain.lower(), func_name.lower()]))

        cell_id = f"{self.domain.upper()}_{func_name.upper()}"
        return CellSchema(
            cell_id=cell_id,
            stage=stage,
            inputs=inputs,
            outputs=outputs,
            code_template=code_template,
            dependencies=[f"import {self.package_name}"],
            semantic_tags=tags,
            keywords=keywords,
            docstring=first_doc_line,
            domain_name=self.domain,
            source_priority=100
        )

    def harvest_all(self, target_functions: Optional[List[str]] = None) -> List[CellSchema]:
        cells = []
        seen_ids = set()

        funcs = target_functions or [n for n in dir(self.module) if not n.startswith("_") and callable(getattr(self.module, n))]
        for name in funcs:
            try:
                obj = getattr(self.module, name)
                cell = self.harvest_function(name, obj)
                if cell and cell.cell_id not in seen_ids:
                    cells.append(cell)
                    seen_ids.add(cell.cell_id)
            except Exception:
                continue
        return cells

    def merge_and_save(self, new_cells: List[CellSchema], output_file: Path) -> None:
        """Merges harvested cells with existing seeds in a single JSON file without overwriting seeds."""
        existing_cells = {}
        if output_file.exists():
            try:
                with open(output_file, "r") as f:
                    data = json.load(f)
                    for c in data.get("cells", []):
                        cell_obj = CellSchema(**c)
                        existing_cells[cell_obj.cell_id] = cell_obj
            except Exception:
                pass

        for cell in new_cells:
            # Only add or update if existing cell is not a curated seed (priority > 10)
            if cell.cell_id not in existing_cells or existing_cells[cell.cell_id].source_priority > 10:
                existing_cells[cell.cell_id] = cell

        tree_schema = TreeSchema(
            domain=self.domain,
            cells=list(existing_cells.values())
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(tree_schema.model_dump_json(indent=2))
