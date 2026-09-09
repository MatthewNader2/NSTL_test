"""
src/universal_harvester.py - Neuro-Symbolic Topological Lattice (NSTL)
Universal Dynamic Library Harvester & Morphism Generator.
Completely domain-agnostic: zero hardcoded library names, types, or keyword heuristics.
"""

from __future__ import annotations
import enum
import importlib
import inspect
import json
import pkgutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import warnings

# Suppress noisy reflection warnings from third-party library namespaces
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from schema import CellSchema, PortSchema, TreeSchema
    from log_config import get_logger
    from tokenizer import CellTokenizer
    from signature_introspector import (
        resolve_signature,
        get_callable_parameters,
        extract_clean_type_name
    )
except ImportError:
    from .schema import CellSchema, PortSchema, TreeSchema
    from .log_config import get_logger
    from .tokenizer import CellTokenizer
    from .signature_introspector import (
        resolve_signature,
        get_callable_parameters,
        extract_clean_type_name
    )

logger = get_logger("universal_harvester")

# Universal language-level base primitives (universal across Python, not specific to any library)
UNIVERSAL_BASE_PRIMITIVES: Set[str] = {
    "int", "float", "str", "bool", "bytes", "bytearray",
    "none", "nonetype", "void", "noreturn", "any", "object",
    "tuple", "list", "dict", "set", "frozenset",
    "iterable", "iterator", "generator", "sequence", "mapping",
    "callable", "type", "ellipsis"
}


def _is_constant_like(attr_name: str, val: Any) -> bool:
    """Universal structural check for constants (PEP 8 convention or Enum instance)."""
    return attr_name.isupper() or isinstance(val, enum.Enum)


def _harvest_constant_cell(
    domain_name: str,
    cid: str,
    m_name: str,
    code_expr: str,
    val: Any,
    tokens: List[str],
    dependencies: List[str],
) -> CellSchema:
    val_type = type(val).__name__
    return CellSchema(
        cell_id=cid,
        stage=1,
        inputs={},
        outputs={"value": PortSchema(type_name=val_type, state="constant")},
        code_template=code_expr,
        dependencies=dependencies,
        semantic_tags=tokens,
        keywords=tokens,
        docstring=f"Constant {code_expr}",
        domain_name=domain_name,
        node_type="constant",
        node_role="constant",
        source_priority=100
    )


def _safe_getattr(obj: Any, attr_name: str) -> Optional[Any]:
    """
    Safely retrieves an attribute. If accessing the attribute triggers a 
    DeprecationWarning or FutureWarning, returns None so NSTL skips the 
    deprecated alias and only harvests the canonical version.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            val = getattr(obj, attr_name, None)
        except Exception:
            return None
        
        # If accessing it triggers a deprecation warning, skip this alias
        if any(issubclass(w.category, (DeprecationWarning, PendingDeprecationWarning, FutureWarning)) for w in caught):
            return None
            
        return val


class UniversalHarvester:
    def __init__(self, domain_name: str, package_name: Optional[str] = None):
        self.domain_name = domain_name
        self.package_name = package_name or domain_name
        try:
            self.root_module = importlib.import_module(self.package_name)
        except ImportError:
            if str(Path.cwd()) not in sys.path:
                sys.path.insert(0, str(Path.cwd()))
            self.root_module = importlib.import_module(self.package_name)
        self.discovered_modules: Dict[str, Any] = {}
        self.domain_types: Set[str] = set()

    def discover_modules(self) -> Dict[str, Any]:
        modules = {self.package_name: self.root_module}
        pkg_path = getattr(self.root_module, "__path__", None)

        def _walk(path, prefix):
            try:
                for item in pkgutil.iter_modules(path, prefix):
                    parts = item.name.split(".")
                    if any(
                        p.lower() in ("tests", "test", "testing", "conftest")
                        or p.lower().startswith("test_")
                        or p.lower().endswith("_test")
                        or (p.startswith("_") and p != f"_{self.package_name}")
                        for p in parts
                    ):
                        continue
                    try:
                        mod = importlib.import_module(item.name)
                        modules[item.name] = mod
                        sub_path = getattr(mod, "__path__", None)
                        if item.ispkg and sub_path:
                            _walk(sub_path, item.name + ".")
                    except (Exception, BaseException):
                        continue
            except (Exception, BaseException):
                pass

        if pkg_path:
            _walk(pkg_path, self.package_name + ".")
        self.discovered_modules = modules
        return modules

    def _discover_domain_type_universe(self) -> Set[str]:
        """
        Pass 1: Discover all domain-specific types T_domain without hardcoding.
        Collects types defined, aliased, or returned by the library that are not
        language-level primitives.
        """
        types: Set[str] = set()

        for m_name, m in self.discovered_modules.items():
            for attr_name in dir(m):
                if attr_name.startswith("_"):
                    continue
                try:
                    obj = _safe_getattr(m, attr_name)
                    if obj is None:
                        continue
                    # 1. Any class declared in the package
                    if inspect.isclass(obj):
                        cls_mod = getattr(obj, "__module__", "") or ""
                        if cls_mod.startswith(self.package_name) or cls_mod.startswith(f"_{self.package_name}"):
                            types.add(obj.__name__)
                    # 2. Any function return type
                    elif callable(obj):
                        sig = resolve_signature(obj, callable_name=attr_name, mod=m)
                        if sig and sig.return_annotation is not inspect.Signature.empty:
                            ret_t = extract_clean_type_name(sig.return_annotation)
                            if ret_t and ret_t.lower() not in UNIVERSAL_BASE_PRIMITIVES:
                                types.add(ret_t)
                except Exception:
                    continue

        # Filter out universal primitives
        self.domain_types = {t for t in types if t.lower() not in UNIVERSAL_BASE_PRIMITIVES}
        return self.domain_types

    def harvest_all(self) -> List[CellSchema]:
        if not self.discovered_modules:
            self.discover_modules()
        if not self.domain_types:
            self._discover_domain_type_universe()

        cells: List[CellSchema] = []
        seen_ids: Set[str] = set()

        # 1. Harvest Constant Morphisms (Stage 1: 0 -> Const)
        for m_name, m in self.discovered_modules.items():
            for attr_name in dir(m):
                if attr_name.startswith("_"):
                    continue
                try:
                    val = _safe_getattr(m, attr_name)
                    if val is None or callable(val) or inspect.isclass(val) or inspect.ismodule(val):
                        continue
                    if not _is_constant_like(attr_name, val):
                        continue
                    cid = f"{self.domain_name.upper()}_{attr_name.upper()}"
                    if cid in seen_ids:
                        continue
                    seen_ids.add(cid)
                    tokens = sorted(list(CellTokenizer.tokenize_identifier(attr_name)))
                    cell = _harvest_constant_cell(
                        self.domain_name, cid, m_name,
                        code_expr=f"{m_name}.{attr_name}",
                        val=val, tokens=tokens,
                        dependencies=[f"import {m_name}"],
                    )
                    cells.append(cell)
                except Exception:
                    continue

        # 2. Harvest Classes: Constructors, Constants, and Methods
        for m_name, m in self.discovered_modules.items():
            for c_name in dir(m):
                if c_name.startswith("_"):
                    continue
                try:
                    cls = _safe_getattr(m, c_name)
                    if not cls or not inspect.isclass(cls):
                        continue
                    cls_mod = getattr(cls, "__module__", "") or ""
                    if not (cls_mod.startswith(self.package_name) or cls_mod.startswith(f"_{self.package_name}")):
                        continue

                    # 2a. Constructor Introspection (Stage 1: 1 -> T_domain)
                    ctor_sig = resolve_signature(
                        cls,
                        callable_name="INIT",
                        parent_cls_name=c_name,
                        mod=m
                    )

                    if ctor_sig is not None:
                        ctor_cid = f"{self.domain_name.upper()}_{c_name.upper()}_INIT"
                        if ctor_cid not in seen_ids:
                            seen_ids.add(ctor_cid)
                            ctor_doc = inspect.getdoc(cls) or f"Construct a new {c_name}"

                            ctor_inputs: Dict[str, PortSchema] = {}
                            required_template_args: List[str] = []

                            for p in ctor_sig.parameters.values():
                                if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                                    continue
                                p_type = extract_clean_type_name(p.annotation)
                                is_req = (p.default is inspect.Parameter.empty)
                                ctor_inputs[p.name] = PortSchema(
                                    type_name=p_type,
                                    state=p.name.lower(),
                                    required=is_req,
                                    description=f"Constructor argument {p.name}"
                                )
                                if is_req:
                                    required_template_args.append(f"{{{p.name}}}")

                            ctor_tokens = sorted(list(CellTokenizer.tokenize_identifier(c_name)))
                            ctor_cell = CellSchema(
                                cell_id=ctor_cid,
                                stage=1,
                                inputs=ctor_inputs,
                                outputs={
                                    "output_data": PortSchema(
                                        type_name=c_name,
                                        state="constructed",
                                        description=f"Newly constructed {c_name}"
                                    )
                                },
                                code_template=f"{{output_var}} = {cls_mod}.{c_name}({', '.join(required_template_args)})",
                                dependencies=[f"import {cls_mod}"],
                                semantic_tags=ctor_tokens,
                                keywords=ctor_tokens,
                                docstring=(ctor_doc.splitlines()[0] if ctor_doc else f"Construct {c_name}"),
                                domain_name=self.domain_name,
                                node_type="constructor",
                                node_role="source",
                                source_priority=50
                            )
                            cells.append(ctor_cell)

                    # 2b. Class-Scoped Constants and Enum Members
                    for attr_name in dir(cls):
                        if attr_name.startswith("_"):
                            continue
                        try:
                            val = getattr(cls, attr_name, None)
                            if val is None or callable(val) or inspect.isclass(val):
                                continue
                            if not _is_constant_like(attr_name, val):
                                continue
                            cattr_cid = f"{self.domain_name.upper()}_{c_name.upper()}_{attr_name.upper()}"
                            if cattr_cid in seen_ids:
                                continue
                            seen_ids.add(cattr_cid)
                            cattr_tokens = sorted(list(
                                CellTokenizer.tokenize_identifier(attr_name)
                                | CellTokenizer.tokenize_identifier(c_name)
                            ))
                            cattr_cell = _harvest_constant_cell(
                                self.domain_name, cattr_cid, m_name,
                                code_expr=f"{cls_mod}.{c_name}.{attr_name}",
                                val=val, tokens=cattr_tokens,
                                dependencies=[f"import {cls_mod}"],
                            )
                            cells.append(cattr_cell)
                        except Exception:
                            continue

                    # 2c. Instance Methods
                    for m_name_attr in dir(cls):
                        if m_name_attr.startswith("_"):
                            continue
                        try:
                            fn = getattr(cls, m_name_attr, None)
                            if not callable(fn) or inspect.isclass(fn):
                                continue

                            cid = f"{self.domain_name.upper()}_{c_name.upper()}_{m_name_attr.upper()}"
                            if cid in seen_ids:
                                continue

                            sig = resolve_signature(
                                fn,
                                callable_name=m_name_attr,
                                parent_cls_name=c_name,
                                mod=m
                            )
                            if sig is None:
                                continue

                            seen_ids.add(cid)
                            doc = inspect.getdoc(fn) or ""
                            first_doc = doc.splitlines()[0] if doc else f"{c_name}.{m_name_attr}"

                            params = list(sig.parameters.values())
                            if params and params[0].name in ("self", "cls"):
                                params = params[1:]

                            inputs: Dict[str, PortSchema] = {
                                "data": PortSchema(
                                    type_name=c_name,
                                    state="any",
                                    required=True,
                                    description=f"Receiver instance of {c_name}"
                                )
                            }
                            required_template_args: List[str] = []

                            for p in params:
                                if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                                    continue
                                p_type = extract_clean_type_name(p.annotation)
                                is_req = (p.default is inspect.Parameter.empty)
                                inputs[p.name] = PortSchema(
                                    type_name=p_type,
                                    state=p.name.lower(),
                                    required=is_req,
                                    description=f"Argument {p.name}"
                                )
                                if is_req:
                                    required_template_args.append(f"{{{p.name}}}")

                            ret_type = extract_clean_type_name(sig.return_annotation)
                            produces_domain = (ret_type in self.domain_types) or (ret_type == c_name)

                            # Structural Morphism Classification (Instance methods always consume c_name)
                            if not produces_domain or ret_type.lower() in ("none", "nonetype", "void", "noreturn"):
                                stage = 3
                                node_type = "sink"
                                node_role = "sink"
                                state = "destination_written"
                                out_type = "str"
                                code_template = f"{{data}}.{m_name_attr}({', '.join(required_template_args)})\n{{output_var}} = 'done'"
                            elif ret_type != c_name:
                                stage = 2
                                node_type = "bridge"
                                node_role = "tunnel"
                                state = "raw"
                                out_type = ret_type
                                code_template = f"{{output_var}} = {{data}}.{m_name_attr}({', '.join(required_template_args)})"
                            else:
                                stage = 2
                                node_type = "function"
                                node_role = "transform"
                                state = "transformed"
                                out_type = c_name
                                code_template = f"{{output_var}} = {{data}}.{m_name_attr}({', '.join(required_template_args)})"

                            outputs = {
                                "output_data": PortSchema(
                                    type_name=out_type,
                                    state=state,
                                    description=f"Output {state}"
                                )
                            }
                            tokens = sorted(list(
                                CellTokenizer.tokenize_identifier(m_name_attr)
                                | CellTokenizer.tokenize_identifier(c_name)
                            ))

                            cell = CellSchema(
                                cell_id=cid,
                                stage=stage,
                                inputs=inputs,
                                outputs=outputs,
                                code_template=code_template,
                                dependencies=[f"import {cls_mod}"],
                                semantic_tags=tokens,
                                keywords=tokens,
                                docstring=first_doc,
                                domain_name=self.domain_name,
                                node_type=node_type,
                                node_role=node_role,
                                source_priority=50
                            )
                            cells.append(cell)
                        except Exception:
                            continue
                except Exception:
                    continue

        # 3. Harvest Module-Level Functions
        for m_name, m in self.discovered_modules.items():
            for attr_name in dir(m):
                if attr_name.startswith("_"):
                    continue
                try:
                    fn = _safe_getattr(m, attr_name)
                    if not fn or not callable(fn) or inspect.isclass(fn):
                        continue
                    fn_mod = getattr(fn, "__module__", None) or m_name
                    if not (fn_mod.startswith(self.package_name) or fn_mod.startswith(f"_{self.package_name}")):
                        continue

                    cid = f"{self.domain_name.upper()}_{attr_name.upper()}"
                    if cid in seen_ids:
                        continue

                    sig = resolve_signature(fn, callable_name=attr_name, mod=m)
                    if sig is None:
                        continue

                    seen_ids.add(cid)
                    doc = inspect.getdoc(fn) or ""
                    first_doc = doc.splitlines()[0] if doc else f"{m_name}.{attr_name}"

                    inputs: Dict[str, PortSchema] = {}
                    required_template_args: List[str] = []
                    required_input_types: List[str] = []

                    for p in sig.parameters.values():
                        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                            continue
                        p_type = extract_clean_type_name(p.annotation)
                        is_req = (p.default is inspect.Parameter.empty)
                        inputs[p.name] = PortSchema(
                            type_name=p_type,
                            state=p.name.lower(),
                            required=is_req,
                            description=f"Argument {p.name}"
                        )
                        if is_req:
                            required_template_args.append(f"{{{p.name}}}")
                            required_input_types.append(p_type)

                    ret_type = extract_clean_type_name(sig.return_annotation)

                    # Pure Structural Monadic Profiling
                    consumes_domain = any(t in self.domain_types for t in required_input_types)
                    produces_domain = (ret_type in self.domain_types)
                    primary_in_type = required_input_types[0] if required_input_types else "any"

                    if consumes_domain and not produces_domain:
                        # D -> 1 (or D -> Base Primitives): SINK
                        stage = 3
                        node_type = "sink"
                        node_role = "sink"
                        state = "destination_written"
                        out_type = "str"
                        code_template = f"{m_name}.{attr_name}({', '.join(required_template_args)})\n{{output_var}} = 'done'"

                    elif not consumes_domain and produces_domain:
                        # 1 -> D (or Primitives -> D): SOURCE
                        stage = 1
                        node_type = "source"
                        node_role = "source"
                        state = "raw"
                        out_type = ret_type
                        code_template = f"{{output_var}} = {m_name}.{attr_name}({', '.join(required_template_args)})"

                    elif consumes_domain and produces_domain:
                        # D1 -> D2: TRANSFORM / BRIDGE
                        stage = 2
                        if primary_in_type != "any" and primary_in_type != ret_type:
                            node_type = "bridge"
                            node_role = "tunnel"
                            state = "raw"
                        else:
                            node_type = "function"
                            node_role = "transform"
                            state = "transformed"
                        out_type = ret_type
                        code_template = f"{{output_var}} = {m_name}.{attr_name}({', '.join(required_template_args)})"

                    else:
                        # Primitives -> Primitives (or fallback utility)
                        stage = 2
                        node_type = "function"
                        node_role = "transform"
                        state = "transformed"
                        out_type = ret_type if ret_type != "any" else "any"
                        code_template = f"{{output_var}} = {m_name}.{attr_name}({', '.join(required_template_args)})"

                    outputs = {
                        "output_data": PortSchema(
                            type_name=out_type,
                            state=state,
                            description=f"Output {state}"
                        )
                    }
                    tokens = sorted(list(CellTokenizer.tokenize_identifier(attr_name)))

                    cell = CellSchema(
                        cell_id=cid,
                        stage=stage,
                        inputs=inputs,
                        outputs=outputs,
                        code_template=code_template,
                        dependencies=[f"import {m_name}"],
                        semantic_tags=tokens,
                        keywords=tokens,
                        docstring=first_doc,
                        domain_name=self.domain_name,
                        node_type=node_type,
                        node_role=node_role,
                        source_priority=50
                    )
                    cells.append(cell)
                except Exception:
                    continue

        return cells

    @classmethod
    def enrich_from_existing_trees(
        cls,
        domain_name: str,
        new_cells: List[CellSchema],
        existing_tree_paths: List[Union[str, Path]]
    ) -> List[CellSchema]:
        old_knowledge: Dict[str, Dict[str, Any]] = {}
        for path in existing_tree_paths:
            p = Path(path)
            if not p.exists():
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                old_cells = data if isinstance(data, list) else data.get("cells", [])
                for oc in old_cells:
                    if isinstance(oc, dict) and "cell_id" in oc:
                        cid = oc["cell_id"].upper()
                        old_knowledge[cid] = oc
                        old_knowledge[cid.replace("_", "")] = oc
            except Exception:
                pass

        for cell in new_cells:
            cid = cell.cell_id.upper()
            suffix = ("_" + cid.split("_", 1)[1]) if "_" in cid else cid
            match = (
                old_knowledge.get(cid)
                or old_knowledge.get(cid.replace("_", ""))
                or next((v for k, v in old_knowledge.items() if k.endswith(suffix)), None)
            )
            if match:
                old_prio = match.get("source_priority", 100)
                if old_prio <= 10:
                    cell.source_priority = old_prio
                old_tags = match.get("semantic_tags", [])
                if old_tags:
                    cell.semantic_tags = list(dict.fromkeys(cell.semantic_tags + old_tags))
                old_kws = match.get("keywords", [])
                if old_kws:
                    cell.keywords = list(dict.fromkeys(cell.keywords + old_kws))
                old_doc = match.get("docstring", "")
                if old_doc and len(old_doc) > len(cell.docstring):
                    cell.docstring = old_doc
                if match.get("verified", False):
                    cell.verified = True
        return new_cells

    def harvest_and_save(self, out_file: Union[str, Path]) -> TreeSchema:
        cells = self.harvest_all()
        out_path = Path(out_file)
        if out_path.exists():
            cells = self.enrich_from_existing_trees(self.domain_name, cells, [out_path])
        tree = TreeSchema(domain=self.domain_name, cells=cells)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(tree.model_dump_json(indent=2))
        logger.info(f"[{self.domain_name}] Successfully harvested {len(cells)} cells to {out_path}")
        return tree


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 universal_harvester.py <library_name>")
        sys.exit(1)
    lib = sys.argv[1]
    harvester = UniversalHarvester(lib)
    out_target = Path("trees") / f"{lib}.json"
    tree = harvester.harvest_and_save(out_target)
    by_archetype: Dict[str, int] = {}
    for c in tree.cells:
        k = f"Stage {c.stage} | Type: {c.node_type} | Role: {c.node_role}"
        by_archetype[k] = by_archetype.get(k, 0) + 1
    print(f"\n[NSTL Universal Harvester] Completed harvest of '{lib}':")
    print(f"Discovered Domain Types: {sorted(list(harvester.domain_types))[:10]}... ({len(harvester.domain_types)} total)")
    print(f"Total cells generated: {len(tree.cells)}")
    print("\nArchetype breakdown:")
    for k, count in sorted(by_archetype.items()):
        print(f"  {k}: {count} nodes")
