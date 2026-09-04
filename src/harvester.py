# src/harvester.py
import json
import inspect
import importlib
import re
import sys
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
try:
    from schema import CellSchema, PortSchema, TreeSchema
except ImportError:
    from .schema import CellSchema, PortSchema, TreeSchema

try:
    from .log_config import get_logger
except ImportError:
    from log_config import get_logger

logger = get_logger('harvester')


def split_identifier_keywords(name: str) -> List[str]:
    """Splits CamelCase, snake_case, and kebab-case into lower-case keywords without dropping characters."""
    # Split CamelCase into tokens
    words = re.findall(r'[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z0-9]|\b)', name)
    if not words:
        words = re.split(r'[_ -]+', name)
    tokens = []
    for w in words:
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', w).lower()
        if len(cleaned) >= 2:
            tokens.append(cleaned)
    return list(dict.fromkeys(tokens))


class IntelligentHarvester:
    """
    Domain-agnostic library harvester. Introspects public APIs across packages and
    submodules, resolves enums/constants, infers typestates and stages, and outputs
    verified CellSchemas with accurate per-submodule dependencies.
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
        1: ["read", "load", "open", "from", "fetch", "create", "imread", "load_audio", "arange", "linspace", "zeros", "ones", "eye", "empty"],
        3: ["write", "save", "export", "dump", "imwrite", "savefig", "save_audio", "show", "display", "to_csv", "to_parquet", "to_json", "to_pickle", "to_excel", "to_hdf", "to_sql", "to_feather", "to_stata"]
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
        "plot": "plotted",
        "scatter": "plotted",
        "bar": "plotted"
    }

    BEHAVIORAL_FLAGS = {
        "inplace", "copy", "axis", "how", "errors", "verbose", "engine",
        "ignore_index", "method", "limit", "downcast", "numeric_only",
        "level", "sort", "observed", "as_index", "group_keys",
        "squeeze", "mangle_dupe_cols", "validate", "kind", "na_position", "drop"
    }

    PRIMARY_OPERAND_NAMES = {
        "value", "val", "fill_value", "to_replace", "by", "on", "dtype",
        "values", "lower", "upper", "expr", "query", "columns", "labels", "target",
        "ascending", "asc"
    }

    def __init__(self, domain: str, package_name: Optional[str] = None):
        self.domain = domain
        self.package_name = package_name or domain
        try:
            self.module = importlib.import_module(self.package_name)
        except ImportError:
            for path in [Path.cwd(), Path.cwd() / "tests" / "fixtures"]:
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            self.module = importlib.import_module(self.package_name)

        self.enum_constants = self._collect_module_constants()

    def _extract_primary_operands(self, callable_obj: Any) -> Tuple[List[str], Dict[str, PortSchema]]:
        """
        Differentiates Behavioral Flags from Primary Operands using reflection.
        Parameters representing primary operands are never stripped even if default is None.
        Returns (template_arg_strs, operand_ports).
        """
        try:
            sig = inspect.signature(callable_obj)
        except (ValueError, TypeError):
            return [], {}

        template_args = []
        operand_ports: Dict[str, PortSchema] = {}

        for p_name, p in sig.parameters.items():
            if p_name in ("self", "cls") or not p_name.isidentifier():
                continue
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if p_name in self.BEHAVIORAL_FLAGS:
                continue

            is_required = (p.default == inspect.Parameter.empty)
            is_operand = (p_name in self.PRIMARY_OPERAND_NAMES) or is_required

            if is_operand:
                type_name = "any"
                state = "any"
                if p_name in ("value", "val", "fill_value", "to_replace", "replacement", "fill"):
                    state = "scalar_literal"
                elif p_name in ("by", "on", "subset", "labels", "columns", "key", "column"):
                    state = "column_name"
                    type_name = "str"
                elif p_name in ("ascending", "asc"):
                    state = "sort_flag"
                    type_name = "bool"
                elif p_name in ("dtype",):
                    state = "dtype_spec"
                    type_name = "str"

                operand_ports[p_name] = PortSchema(
                    type_name=type_name,
                    state=state,
                    description=f"Operand {p_name}",
                    required=is_required
                )
                if is_required:
                    template_args.append(f"{{{p_name}}}")
                else:
                    template_args.append(f"{p_name}={{{p_name}}}")

        return template_args, operand_ports

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
            if v == name_lower or name_lower.startswith(v) or any(p in ("filepath", "filename", "path", "uri", "src_path") for p in params):
                return 1
        for v in self.STAGE_VERBS[3]:
            if v == name_lower or name_lower.startswith(v) or any(p in ("dest_path", "save_path", "out_path", "output_path", "filename") and "read" not in name_lower and "load" not in name_lower for p in params):
                return 3
        return 2

    def _infer_typestate(self, func_name: str, default_state: str = "transformed") -> str:
        name_lower = func_name.lower()
        for verb, state in self.STATE_INFERENCES.items():
            if verb in name_lower:
                return state
        return default_state

    def _resolve_module_and_dep(self, func_name: str, func_obj: Any, parent_mod_name: Optional[str] = None) -> Tuple[str, List[str], str]:
        """
        Resolves canonical submodule name, exact import dependency, and execution prefix.
        Returns: (canonical_module, dependencies_list, call_prefix)
        """
        raw_mod = getattr(func_obj, "__module__", None) or parent_mod_name or self.package_name
        
        # Clean private paths (e.g. sklearn.linear_model._coordinate_descent -> sklearn.linear_model)
        parts = raw_mod.split(".")
        clean_parts = [p for p in parts if not p.startswith("_")]
        clean_mod = ".".join(clean_parts) if clean_parts else raw_mod

        if inspect.isclass(func_obj):
            dependencies = [f"from {clean_mod} import {func_name}"]
            call_prefix = func_name
        elif clean_mod == self.package_name:
            dependencies = [f"import {self.package_name}"]
            call_prefix = f"{self.package_name}.{func_name}"
        else:
            dependencies = [f"import {clean_mod}"]
            call_prefix = f"{clean_mod}.{func_name}"

        return clean_mod, dependencies, call_prefix

    def harvest_function(self, func_name: str, func_obj: Any, parent_mod_name: Optional[str] = None) -> Optional[CellSchema]:
        """Introspects a single callable and produces a structured CellSchema."""
        if not callable(func_obj) or func_name.startswith("_"):
            return None

        doc = inspect.getdoc(func_obj) or getattr(func_obj, "__doc__", "") or ""
        first_doc_line = doc.split("\n")[0].strip() if doc else f"{func_name} operation"

        try:
            sig = inspect.signature(func_obj)
        except (ValueError, TypeError):
            sig = None

        default_container = self.CONTAINER_TYPES.get(self.domain, self.CONTAINER_TYPES.get(self.package_name, "DataObject"))
        params = list(sig.parameters.keys()) if sig else ["data"]
        stage = self._infer_stage(func_name, params, default_container)
        out_state = self._infer_typestate(func_name, "raw" if stage == 1 else "processed")

        clean_mod, dependencies, call_prefix = self._resolve_module_and_dep(func_name, func_obj, parent_mod_name)

        inputs: Dict[str, PortSchema] = {}
        outputs: Dict[str, PortSchema] = {}
        template_args = []

        if stage == 1:
            inputs["filepath"] = PortSchema(type_name="str", state="source_identifier", description="Input file path")
            template_args.append("{filepath}")
            outputs["output_data"] = PortSchema(type_name=default_container, state="raw")
        elif stage == 3:
            inputs["data"] = PortSchema(type_name=default_container, state="any")
            inputs["dest_path"] = PortSchema(type_name="str", state="dest_identifier", description="Destination file path")
            if any(p in ("filename", "dest_path", "save_path", "output_path", "file") for p in params):
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


        # Capability-driven template construction via runtime introspection.
        # No domain-specific if/elif chains — uses hasattr/inspect to detect
        # method vs function, estimator vs plain class, etc.

        # Detect if func_name is an instance method on the domain's container class
        container_cls = self._resolve_container_class()
        is_instance_method = container_cls is not None and hasattr(container_cls, func_name)

        if is_instance_method:
            # Instance method on the domain's primary container (e.g. DataFrame.dropna, Mat.reshape)
            if stage == 1:
                code_template = f"{{output_var}} = {call_prefix}({{filepath}})"
            elif stage == 3:
                code_template = f"{{data}}.{func_name}({{dest_path}})\n{{output_var}} = {{dest_path}}"
            else:
                arg_strs, op_ports = self._extract_primary_operands(func_obj)
                inputs.update(op_ports)
                args_joined = f"({', '.join(arg_strs)})" if arg_strs else "()"
                code_template = f"{{output_var}} = {{data}}.{func_name}{args_joined}"
        elif inspect.isclass(func_obj):
            # Class instantiation — detect capabilities via introspection
            if hasattr(func_obj, "fit_transform"):
                code_template = f"{{output_var}} = {func_name}().fit_transform({', '.join(template_args)})"
            elif hasattr(func_obj, "fit") or hasattr(func_obj, "predict"):
                code_template = f"{{output_var}} = {func_name}().fit({', '.join(template_args)})"
            else:
                code_template = f"{{output_var}} = {func_name}({', '.join(template_args)})"
        elif hasattr(func_obj, "rvs"):
            # Statistical distribution object (scipy.stats etc.)
            code_template = f"{{output_var}} = {call_prefix}.rvs(size=100)"
        elif stage == 3:
            # Generic Stage 3 (egress/export): call with data + dest
            code_template = f"{{data}}.{func_name}({', '.join(template_args[1:])})\n{{output_var}} = {{dest_path}}"
        else:
            # Generic function call — fully domain-agnostic
            code_template = f"{{output_var}} = {call_prefix}({', '.join(template_args)})"

        # Semantic Tags & Keywords using lossless word-boundary tokenization
        split_words = split_identifier_keywords(func_name)
        tags = list(dict.fromkeys([func_name.lower(), self.domain.lower(), out_state] + split_words))
        keywords = list(dict.fromkeys(tags + [self.domain.lower(), func_name.lower()]))

        # Format clean cell_id
        if "_" in func_name:
            cell_id_suffix = func_name.upper()
        else:
            cell_id_suffix = "_".join([w.upper() for w in split_words])
            if cell_id_suffix == "MIN_MAX_SCALER":
                cell_id_suffix = "MINMAX_SCALER"

        cell_id = f"{self.domain.upper()}_{cell_id_suffix}"
        return CellSchema(
            cell_id=cell_id,
            stage=stage,
            inputs=inputs,
            outputs=outputs,
            code_template=code_template,
            dependencies=dependencies,
            semantic_tags=tags,
            keywords=keywords,
            docstring=first_doc_line,
            domain_name=self.domain,
            source_priority=100
        )

    def harvest_method(self, class_name: str, cls: Any, method_name: str, method_obj: Any, parent_mod_name: Optional[str] = None) -> Optional[CellSchema]:
        """Introspects a single public instance method on a class."""
        if not callable(method_obj) or method_name.startswith("_"):
            return None

        # Exclude base object methods
        if method_name in ("mro", "count", "index") and cls is object:
            return None

        doc = inspect.getdoc(method_obj) or getattr(method_obj, "__doc__", "") or ""
        first_doc_line = doc.split("\n")[0].strip() if doc else f"{class_name}.{method_name} method"

        try:
            sig = inspect.signature(method_obj)
        except (ValueError, TypeError):
            sig = None

        default_container = self.CONTAINER_TYPES.get(self.domain, self.CONTAINER_TYPES.get(self.package_name, "DataObject"))
        params = [p for p in sig.parameters.keys() if p != "self"] if sig else ["data"]
        stage = self._infer_stage(method_name, params, default_container)
        out_state = self._infer_typestate(method_name, "raw" if stage == 1 else "processed")

        clean_mod, dependencies, _ = self._resolve_module_and_dep(class_name, cls, parent_mod_name)

        inputs: Dict[str, PortSchema] = {}
        outputs: Dict[str, PortSchema] = {}

        # Capability-driven method template construction — no domain-specific if/elif chains.
        # Detect container methods vs estimator methods vs generic instance methods via introspection.
        container_cls = self._resolve_container_class()
        is_container_method = container_cls is not None and (
            class_name == container_cls.__name__
            or (hasattr(container_cls, '__mro__') and class_name in [c.__name__ for c in container_cls.__mro__])
        )

        if is_container_method:
            # Method on the domain's primary container (e.g. DataFrame.dropna, Series.map)
            inputs["data"] = PortSchema(type_name=default_container, state="any", description=f"Input {class_name}", required=True)
            if stage == 3:
                inputs["dest_path"] = PortSchema(type_name="str", state="dest_identifier", description="Destination file path", required=True)
                code_template = f"{{data}}.{method_name}({{dest_path}})\n{{output_var}} = {{dest_path}}"
                outputs["output_data"] = PortSchema(type_name="str", state="filepath_written", required=True)
            else:
                arg_strs, op_ports = self._extract_primary_operands(method_obj)
                inputs.update(op_ports)
                args_joined = f"({', '.join(arg_strs)})" if arg_strs else "()"
                code_template = f"{{output_var}} = {{data}}.{method_name}{args_joined}"
                outputs["output_data"] = PortSchema(type_name=default_container, state=out_state, required=True)
        elif hasattr(cls, "fit") or hasattr(cls, "predict") or hasattr(cls, "fit_transform"):
            # Estimator/model-like class (detected via fit/predict/fit_transform capabilities)
            inputs["model"] = PortSchema(type_name=class_name, state="any", description=f"{class_name} estimator", required=True)
            inputs["data"] = PortSchema(type_name=default_container, state="any", description="Input data", required=True)
            code_template = f"{{output_var}} = {{model}}.{method_name}({{data}})"
            outputs["output_data"] = PortSchema(type_name=default_container, state=out_state, required=True)
            dependencies = [f"from {clean_mod} import {class_name}"]
        else:
            # Generic class instance method
            inputs["instance"] = PortSchema(type_name=class_name, state="any", description=f"{class_name} instance", required=True)
            inputs["data"] = PortSchema(type_name=default_container, state="any", description="Input data", required=True)
            code_template = f"{{output_var}} = {{instance}}.{method_name}({{data}})"
            outputs["output_data"] = PortSchema(type_name=default_container, state=out_state, required=True)
            dependencies = [f"from {clean_mod} import {class_name}"]

        split_class = split_identifier_keywords(class_name)
        split_method = split_identifier_keywords(method_name)
        tags = list(dict.fromkeys([class_name.lower(), method_name.lower(), self.domain.lower(), out_state] + split_class + split_method))
        keywords = list(dict.fromkeys(tags + [self.domain.lower(), class_name.lower(), method_name.lower()]))

        cell_id = f"{self.domain.upper()}_{class_name.upper()}_{method_name.upper()}"
        return CellSchema(
            cell_id=cell_id,
            stage=stage,
            inputs=inputs,
            outputs=outputs,
            code_template=code_template,
            dependencies=dependencies,
            semantic_tags=tags,
            keywords=keywords,
            docstring=first_doc_line,
            domain_name=self.domain,
            source_priority=100
        )

    def harvest_all(self, target_functions: Optional[List[str]] = None) -> List[CellSchema]:
        cells = []
        seen_ids = set()

        funcs = target_functions or [n for n in dir(self.module) if not n.startswith("_") and callable(getattr(self.module, n, None))]
        for name in funcs:
            try:
                obj = getattr(self.module, name, None)
                if obj is None:
                    continue
                cell = self.harvest_function(name, obj, parent_mod_name=self.package_name)
                if cell and cell.cell_id not in seen_ids:
                    cells.append(cell)
                    seen_ids.add(cell.cell_id)

                if inspect.isclass(obj):
                    for m_name in dir(obj):
                        if m_name.startswith("_"):
                            continue
                        try:
                            m_obj = getattr(obj, m_name, None)
                            if m_obj is not None and callable(m_obj):
                                m_cell = self.harvest_method(name, obj, m_name, m_obj, parent_mod_name=self.package_name)
                                if m_cell and m_cell.cell_id not in seen_ids:
                                    cells.append(m_cell)
                                    seen_ids.add(m_cell.cell_id)
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"[HARVESTER] Failed to harvest '{name}': {e}")
                continue
        return cells

    def harvest_modules(self, module_names: List[str]) -> List[CellSchema]:
        """Harvests public callables and class instance methods across multiple submodules."""
        cells = []
        seen_ids = set()

        for mod_name in module_names:
            try:
                mod = importlib.import_module(mod_name)
            except ImportError as e:
                logger.warning(f"[HARVESTER] Could not import module '{mod_name}': {e}")
                continue

            for name in dir(mod):
                if name.startswith("_"):
                    continue
                try:
                    obj = getattr(mod, name, None)
                    if obj is None or not callable(obj):
                        continue
                    cell = self.harvest_function(name, obj, parent_mod_name=mod_name)
                    if cell and cell.cell_id not in seen_ids:
                        cells.append(cell)
                        seen_ids.add(cell.cell_id)

                    if inspect.isclass(obj):
                        for m_name in dir(obj):
                            if m_name.startswith("_"):
                                continue
                            try:
                                m_obj = getattr(obj, m_name, None)
                                if m_obj is not None and callable(m_obj):
                                    m_cell = self.harvest_method(name, obj, m_name, m_obj, parent_mod_name=mod_name)
                                    if m_cell and m_cell.cell_id not in seen_ids:
                                        cells.append(m_cell)
                                        seen_ids.add(m_cell.cell_id)
                            except Exception:
                                pass
                except Exception as e:
                    logger.warning(f"[HARVESTER] Error harvesting '{name}' from '{mod_name}': {e}")
                    continue

        return cells

    def merge_and_save(self, new_cells: List[CellSchema], output_file: Path) -> None:
        """Merges harvested cells with existing seeds in a single JSON file without overwriting curated seeds."""
        existing_cells = {}
        if output_file.exists():
            try:
                with open(output_file, "r") as f:
                    data = json.load(f)
                    for c in data.get("cells", []):
                        cell_obj = CellSchema(**c)
                        existing_cells[cell_obj.cell_id] = cell_obj
            except Exception as e:
                logger.warning(f"[HARVESTER] Failed to load existing cells from {output_file}: {e}")

        for cell in new_cells:
            # Only add or update if existing cell is not a curated seed (priority > 10)
            if cell.cell_id not in existing_cells:
                existing_cells[cell.cell_id] = cell
            elif existing_cells[cell.cell_id].source_priority > 10:
                # Retain existing docstring if already docs-enriched
                if existing_cells[cell.cell_id].docstring and existing_cells[cell.cell_id].enrichment_source == "docs":
                    cell.docstring = existing_cells[cell.cell_id].docstring
                    cell.enrichment_source = "docs"
                    cell.enriched_at = existing_cells[cell.cell_id].enriched_at
                existing_cells[cell.cell_id] = cell

        tree_schema = TreeSchema(
            domain=self.domain,
            cells=list(existing_cells.values())
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(tree_schema.model_dump_json(indent=2))
