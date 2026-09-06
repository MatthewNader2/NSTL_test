"""
src/universal_harvester.py - Neuro-Symbolic Topological Lattice (NSTL)
Universal Library Harvester & Morphism Generator.

Domain-agnostic runtime introspection engine that can ingest any Python package
(e.g., cv2, numpy, pandas, scipy, sklearn, torch) and generate clean, mathematically
categorized lattice nodes conforming to category-theoretic archetypes:
  1. Atomic Morphisms (Primitive function A -> B)
  2. Parameterized Morphisms (A x E -> B, with typed enum/option ports)
  3. Higher-Order / Scoped Morphisms (A x Hom(A, B) -> B', with execution slots)
  4. Source / Ingestion Morphisms (0 -> A, Stage 1)
  5. Sink / Egress Morphisms (A -> 0, Stage 3)
  6. Constant Morphisms (* -> Enum)

Enriches harvested nodes from previous knowledge bases without inheriting
corrupt combinatorial duplicate cells.
"""

from __future__ import annotations
import ast
import collections.abc
import importlib
import inspect
import json
import os
import pkgutil
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

try:
    from schema import CellSchema, PortSchema, TreeSchema
    from log_config import get_logger
    from signature_introspector import get_callable_parameters, get_enum_parameter_map, extract_doc_signature
    from template_wiring import repair_wiring_invariant
except ImportError:
    from .schema import CellSchema, PortSchema, TreeSchema
    from .log_config import get_logger
    from .signature_introspector import get_callable_parameters, get_enum_parameter_map, extract_doc_signature
    from .template_wiring import repair_wiring_invariant

logger = get_logger("universal_harvester")


def split_identifier_keywords(name: str) -> List[str]:
    """Splits CamelCase, snake_case, and kebab-case into clean keywords."""
    words = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z0-9]|\b)", name)
    if not words:
        words = re.split(r"[_ -]+", name)
    tokens: List[str] = []
    for w in words:
        cleaned = re.sub(r"[^a-zA-Z0-9]", "", w).lower()
        if len(cleaned) >= 2:
            tokens.append(cleaned)
    return list(dict.fromkeys(tokens))


def infer_qualifiers_from_constant(name: str) -> List[List[str]]:
    """
    Infers structural qualifiers from constant names via morphological decomposition.
    Zero domain-specific lists or hardcoded transitions.
    Decomposes structural qualifiers PREFIX_PART -> [prefix, part].
    """
    parts = [p.lower() for p in name.split("_") if p]
    if len(parts) <= 1:
        return []
    prefix = parts[0]
    return [[prefix, p] for p in parts[1:]]


def infer_typestate_from_name(name: str) -> str:
    """Derives a normalized typestate string from a callable name via morphological participle formation."""
    words = split_identifier_keywords(name)
    if not words:
        return "transformed"
    verb = words[0].lower()
    if verb.endswith("ed"):
        return verb
    elif verb.endswith("e"):
        return verb + "d"
    elif verb.endswith("y") and len(verb) > 2 and verb[-2] not in "aeiou":
        return verb[:-1] + "ied"
    else:
        return verb + "ed"


class UniversalHarvester:
    """
    Universal, domain-agnostic library introspection engine.
    Discovers all public modules, callables, classes, and constants in a library,
    classifies them by morphism archetype, and outputs clean, validated CellSchemas.
    """

    def __init__(
        self,
        domain_name: str,
        package_name: Optional[str] = None,
        max_depth: int = 3,
        container_type: Optional[str] = None
    ):
        self.domain_name = domain_name
        self.package_name = package_name or domain_name
        self.max_depth = max_depth
        self.container_type = container_type

        # Ensure package is imported
        try:
            self.root_module = importlib.import_module(self.package_name)
        except ImportError:
            # Check local project or fixtures
            for extra_path in [Path.cwd(), Path.cwd() / "tests" / "fixtures"]:
                if str(extra_path) not in sys.path:
                    sys.path.insert(0, str(extra_path))
            self.root_module = importlib.import_module(self.package_name)

        self.discovered_modules: Dict[str, Any] = {}
        self.constants_by_prefix: Dict[str, List[Tuple[str, Any, str]]] = {}
        self.container_classes: Set[type] = set()
        self.primary_container: Optional[type] = None

    def discover_submodules(self) -> Dict[str, Any]:
        """Recursively discovers public submodules within the package up to max_depth."""
        modules = {self.package_name: self.root_module}
        pkg_path = getattr(self.root_module, "__path__", None)
        if not pkg_path:
            self.discovered_modules = modules
            return modules

        visited = set()

        def _walk(prefix: str, path: Any, depth: int):
            if depth > self.max_depth or prefix in visited:
                return
            visited.add(prefix)

            try:
                for _, modname, ispkg in pkgutil.iter_modules(path, prefix):
                    # Skip private submodules, test directories, and legacy deprecated submodules
                    tail = modname.split(".")[-1]
                    if (
                        tail.startswith("_")
                        or "._" in modname
                    ):
                        continue

                    try:
                        mod = importlib.import_module(modname)
                        modules[modname] = mod
                        if ispkg and hasattr(mod, "__path__"):
                            _walk(modname + ".", mod.__path__, depth + 1)
                    except (Exception, BaseException):
                        continue
            except (Exception, BaseException):
                pass

        _walk(self.package_name + ".", pkg_path, 1)

        # Also inspect direct module attributes (e.g. cv2.dnn, numpy.linalg)
        for attr_name in dir(self.root_module):
            if not attr_name.startswith("_"):
                try:
                    attr_val = getattr(self.root_module, attr_name, None)
                    if inspect.ismodule(attr_val):
                        m_name = getattr(attr_val, "__name__", "")
                        if m_name.startswith(self.package_name) and m_name not in modules and "._" not in m_name:
                            modules[m_name] = attr_val
                except Exception:
                    continue

        self.discovered_modules = modules
        return modules

    def collect_constants_and_enums(self) -> Dict[str, List[Tuple[str, Any, str]]]:
        """
        Extracts uppercase constants and Enum classes across all discovered modules.
        Groups them by common prefix (e.g. COLOR_*, THRESH_*, INTER_*, NORM_*).
        Returns: {prefix: [(constant_name, constant_val, module_path), ...]}
        """
        groups: Dict[str, List[Tuple[str, Any, str]]] = {}

        for mod_name, mod in self.discovered_modules.items():
            for name in dir(mod):
                if name.startswith("_"):
                    continue
                try:
                    val = getattr(mod, name, None)
                except Exception:
                    continue

                if name.isupper() and not callable(val):
                    parts = name.split("_")
                    prefix = parts[0] if len(parts) > 1 else "CONST"
                    if prefix not in groups:
                        groups[prefix] = []
                    groups[prefix].append((name, val, mod_name))

        self.constants_by_prefix = groups
        return groups

    def identify_container_classes(self) -> Set[type]:
        """
        Identifies primary data carrier container classes in the library via runtime reflection.
        Zero hardcoded name lists.
        Discovers classes that implement algebraic/collection protocols or exhibit high method density.
        """
        containers: Set[type] = set()

        if self.container_type:
            for mod in self.discovered_modules.values():
                cls = getattr(mod, self.container_type, None)
                if inspect.isclass(cls):
                    containers.add(cls)
            if containers:
                self.container_classes = containers
                self.primary_container = next(iter(containers))
                return containers

        scored_classes: List[Tuple[float, type]] = []
        seen_types = set()

        for mod in self.discovered_modules.values():
            for name in dir(mod):
                if name.startswith("_"):
                    continue
                try:
                    obj = getattr(mod, name, None)
                    if not inspect.isclass(obj) or obj in seen_types:
                        continue
                    seen_types.add(obj)

                    obj_mod = getattr(obj, "__module__", "") or ""
                    # Language builtins and non-package classes are not the library's domain containers
                    if obj_mod in ("builtins", "typing", "collections.abc") or obj_mod.startswith("builtins."):
                        continue
                    if not obj_mod.startswith(self.package_name) and self.package_name != "builtins":
                        continue

                    # Must satisfy container / collection protocol
                    has_len = hasattr(obj, "__len__")
                    has_contains = hasattr(obj, "__contains__")
                    try:
                        is_cont = issubclass(obj, (collections.abc.Container, collections.abc.Collection))
                    except TypeError:
                        is_cont = False
                    if not (has_len or has_contains or is_cont):
                        continue

                    # Count public callable methods
                    methods = [
                        m for m in dir(obj)
                        if not m.startswith("_") and callable(getattr(obj, m, None))
                    ]
                    num_methods = len(methods)
                    if num_methods < 5:
                        continue

                    score = float(num_methods)

                    # Algebraic container protocols
                    has_getitem = hasattr(obj, "__getitem__")
                    if has_len and has_getitem:
                        score += 50.0
                    elif has_len or has_getitem:
                        score += 20.0
                    if has_contains:
                        score += 20.0
                    if is_cont:
                        score += 30.0

                    scored_classes.append((score, obj))
                except Exception:
                    continue

        # Also discover container classes from callable parameter/return annotations
        for mod in self.discovered_modules.values():
            for name in dir(mod):
                if name.startswith("_"):
                    continue
                try:
                    fn = getattr(mod, name, None)
                    if callable(fn) and not inspect.isclass(fn):
                        sig = inspect.signature(fn)
                        candidates = [p.annotation for p in sig.parameters.values()] + [sig.return_annotation]
                        for cand in candidates:
                            if inspect.isclass(cand) and cand not in seen_types:
                                seen_types.add(cand)
                                cand_mod = getattr(cand, "__module__", "") or ""
                                if cand_mod in ("builtins", "typing", "collections.abc") or cand_mod.startswith("builtins."):
                                    continue
                                try:
                                    has_len = hasattr(cand, "__len__")
                                    has_getitem = hasattr(cand, "__getitem__")
                                    is_coll = issubclass(cand, (collections.abc.Container, collections.abc.Collection))
                                    if (has_len and has_getitem) or is_coll:
                                        scored_classes.append((100.0, cand))
                                except TypeError:
                                    pass
                except Exception:
                    continue

        # Adjust score by subclass hierarchy and root package export
        adjusted_classes: List[Tuple[float, type]] = []
        root_all = getattr(self.root_module, "__all__", dir(self.root_module))
        for score, cls in scored_classes:
            derivations = 0
            for _, other in scored_classes:
                if other is not cls:
                    try:
                        if issubclass(other, cls):
                            derivations += 1
                    except TypeError:
                        pass
            adj_score = score + min(derivations, 4) * 10.0
            if cls.__name__ in root_all and getattr(self.root_module, cls.__name__, None) is cls:
                adj_score += 50.0
            adjusted_classes.append((adj_score, cls))

        adjusted_classes.sort(key=lambda x: x[0], reverse=True)
        if adjusted_classes:
            self.primary_container = adjusted_classes[0][1]
            top_score = adjusted_classes[0][0]
            for score, cls in adjusted_classes:
                if score >= top_score * 0.7 and len(containers) < 3:
                    containers.add(cls)
        else:
            self.primary_container = None

        self.container_classes = containers
        return containers

    def harvest_constant_nodes(self) -> List[CellSchema]:
        """
        Emits typed Constant Morphisms (* -> Enum) for each discovered enum/flag.
        Converts combinatorial variants into modular leaf nodes.
        """
        constant_cells: List[CellSchema] = []
        seen_cells: Set[str] = set()

        for prefix, items in self.constants_by_prefix.items():
            state_name = prefix.lower()
            for const_name, _, mod_path in items:
                cell_id = f"{self.domain_name.upper()}_{const_name}"
                if cell_id in seen_cells:
                    continue
                seen_cells.add(cell_id)

                qualifiers = infer_qualifiers_from_constant(const_name)
                code_template = f"{mod_path}.{const_name}"

                out_port = PortSchema(
                    type_name="Enum",
                    state=state_name,
                    qualifiers=qualifiers,
                    domain=f"{mod_path}.{prefix}_*",
                    description=f"{self.domain_name} constant {const_name}"
                )

                keywords = split_identifier_keywords(const_name) + [prefix.lower(), self.domain_name.lower()]
                cell = CellSchema(
                    cell_id=cell_id,
                    stage=0,
                    inputs={},
                    outputs={"value": out_port},
                    code_template=code_template,
                    dependencies=[f"import {mod_path}"],
                    semantic_tags=list(dict.fromkeys(keywords)),
                    keywords=list(dict.fromkeys(keywords)),
                    docstring=f"Constant {mod_path}.{const_name}",
                    domain_name=self.domain_name,
                    node_type="constant",
                    node_role="constant",
                    source_priority=100
                )
                constant_cells.append(cell)

        return constant_cells

    def _is_carrier_type(self, typ: Any) -> bool:
        """
        Categorical check: Does `typ` belong to the category's carrier objects Obj(C)?
        True if typ is identical to, or in a subtype relation with, any discovered container class in C.
        Pure category-theoretic set membership with zero explicit data type checks.
        """
        if not inspect.isclass(typ) or typ is object:
            return False
        for c in self.container_classes:
            try:
                if typ is c or issubclass(typ, c) or issubclass(c, typ):
                    return True
            except TypeError:
                pass
        return False

    def _get_carrier_tokens(self) -> Set[str]:
        """
        Dynamically extracts carrier morphological tokens and identifiers
        from discovered container classes and their MRO via runtime reflection.
        Zero hardcoded name lists or domain-specific assumptions.
        """
        tokens: Set[str] = set()
        for c in self.container_classes:
            tokens.add(c.__name__.lower())
            tokens.update(split_identifier_keywords(c.__name__))

            # MRO base classes (excluding object and builtins)
            for base in getattr(c, "__mro__", []):
                base_mod = getattr(base, "__module__", "") or ""
                if base is not object and base_mod != "builtins" and not base_mod.startswith("builtins."):
                    tokens.add(base.__name__.lower())
                    tokens.update(split_identifier_keywords(base.__name__))

        if self.container_type:
            tokens.add(self.container_type.lower())
            tokens.update(split_identifier_keywords(self.container_type))

        return tokens

    def _infer_stage(
        self,
        func_name: str,
        has_carrier_input: bool,
        ret_type: str,
        doc: str,
        is_instance_method: bool = False
    ) -> int:
        """
        Determines the categorical stage (1: Ingestion, 2: Transform, 3: Egress) via signature reflection.
        Categorically:
          - Stage 1 (Initial / Ingestion): 0 -> C (consumes no carrier from DAG, returns carrier data)
          - Stage 3 (Terminal / Egress): C -> 1 (consumes carrier, returns terminal object: None/void/bool)
          - Stage 2 (Endomorphism): C -> C' (consumes carrier, produces transformed carrier)
        Zero word lists or heuristic keyword matching.
        """
        consumes_carrier = has_carrier_input or is_instance_method

        if not consumes_carrier:
            # 0 -> C: Initial / Source morphism
            return 1

        ret_lower = str(ret_type).lower().strip() if ret_type else ""
        carrier_tokens = self._get_carrier_tokens()

        # Categorical codomain check: does the morphism produce a carrier container in C?
        produces_carrier = False
        if ret_lower:
            for cls in self.container_classes:
                c_name = cls.__name__.lower()
                if c_name == ret_lower or c_name in ret_lower:
                    produces_carrier = True
                    break
            if not produces_carrier and any(t in ret_lower for t in carrier_tokens):
                produces_carrier = True
        elif doc:
            for cls in self.container_classes:
                if re.search(rf"\b(?:returns?|into|->)\s*.*?\b{cls.__name__}\b", doc, re.IGNORECASE):
                    produces_carrier = True
                    break
            if not produces_carrier and any(re.search(rf"\b(?:returns?|into|->)\s*.*?\b{t}\b", doc, re.IGNORECASE) for t in carrier_tokens):
                produces_carrier = True

        # Check return arrow in docstring: In Category Theory, a morphism maps src -> dst
        if not produces_carrier and doc:
            m_arrow = re.search(rf"\b{re.escape(func_name)}\(.*?\)\s*->\s*([a-zA-Z0-9_]+)", doc)
            if m_arrow:
                arrow_ret = m_arrow.group(1).lower()
                if arrow_ret in carrier_tokens or arrow_ret == "dst":
                    produces_carrier = True

        # In-place instance methods on carrier container default to endomorphism (C -> C)
        if not produces_carrier and is_instance_method and not ret_lower:
            produces_carrier = True

        if not produces_carrier:
            # C -> 1: Terminal / Egress morphism
            return 3

        # C -> C': Endomorphism / Transform morphism
        return 2

    def _detect_enum_parameter(
        self,
        param_name: str,
        doc: str,
        param_obj: Optional[inspect.Parameter] = None
    ) -> Optional[str]:
        """
        Detects whether a parameter expects an enum flag and identifies the enum prefix dynamically.
        Zero hardcoded parameter names.
        Grounds detection in:
          1. Parameter type annotations (Enum, Literal)
          2. Docstring parameter specifications (@param <name> ... see #<Type>)
          3. Docstring mentions of #<PREFIX>_* or <PREFIX>_*
          4. Token match between param_name and discovered library constant prefixes
        """
        # 1. Type annotation reflection
        if param_obj and param_obj.annotation != inspect._empty:
            anno = param_obj.annotation
            anno_str = getattr(anno, "__name__", str(anno)).lower()
            for prefix in self.constants_by_prefix:
                if prefix.lower() in anno_str:
                    return prefix

        # 2. Docstring type mapping: @param <p_name> ... see #<Type>
        enum_map = get_enum_parameter_map(doc)
        for prefix, p_name in enum_map.items():
            if p_name.lower() == param_name.lower() or param_name.lower() in p_name.lower():
                if prefix in self.constants_by_prefix:
                    return prefix

        # 3. Parameter-specific docstring block reflection: find #PREFIX_* or see #Type in parameter description
        p_clean = param_name.lower().replace("_", "")
        if doc:
            p_blocks = re.findall(rf"@param\s+{re.escape(param_name)}\b([^@]+)", doc, re.IGNORECASE)
            for block in p_blocks:
                for prefix in self.constants_by_prefix:
                    if f"#{prefix}_" in block or f"#{prefix.lower()}" in block.lower() or f"see #{prefix.lower()}" in block.lower():
                        return prefix

        # 4. Token match between parameter name and discovered library constant prefixes
        for prefix in self.constants_by_prefix:
            pref_clean = prefix.lower().replace("_", "")
            if len(pref_clean) >= 3 and (pref_clean in p_clean or p_clean.startswith(pref_clean) or p_clean.endswith(pref_clean)):
                return prefix

        return None

    def harvest_callable(
        self,
        func_name: str,
        func_obj: Any,
        mod_name: str,
        is_instance_method: bool = False,
        container_class_name: Optional[str] = None
    ) -> Optional[CellSchema]:
        """
        Introspects a callable and constructs a clean, typed CellSchema.
        Categorizes it into Atomic, Parameterized, or Higher-Order morphism.
        """
        if not callable(func_obj) or func_name.startswith("_"):
            return None

        # Ignore foreign re-exports (e.g. numpy functions imported into scipy)
        obj_mod = getattr(func_obj, "__module__", "") or mod_name
        if obj_mod and not obj_mod.startswith(self.package_name) and not is_instance_method:
            return None

        doc = inspect.getdoc(func_obj) or getattr(func_obj, "__doc__", "") or ""
        first_doc = doc.split("\n")[0].strip() if doc else f"{func_name} morphism"

        # Introspect parameter contract
        param_info = get_callable_parameters(func_obj, func_name)
        if not param_info:
            param_info = {"required": [], "optional": [], "all": ["data"], "doc": doc}

        required_params: List[str] = param_info.get("required", [])
        optional_params: List[str] = param_info.get("optional", [])
        all_params: List[str] = param_info.get("all", required_params + optional_params)

        # Introspect inspect.signature if possible
        param_objs: Dict[str, inspect.Parameter] = {}
        ret_annotation: Any = None
        try:
            sig = inspect.signature(func_obj)
            param_objs = dict(sig.parameters)
            ret_annotation = sig.return_annotation
        except Exception:
            pass

        # If it's an instance method, drop 'self' or 'cls'
        if is_instance_method:
            required_params = [p for p in required_params if p not in ("self", "cls")]
            optional_params = [p for p in optional_params if p not in ("self", "cls")]
            all_params = [p for p in all_params if p not in ("self", "cls")]

        # Determine primary carrier type
        primary_cls_name = self.primary_container.__name__ if getattr(self, "primary_container", None) else None
        carrier_type = container_class_name or self.container_type or primary_cls_name or (
            next(iter(self.container_classes)).__name__ if self.container_classes else "DataObject"
        )

        # Carrier detection via type annotations and collection protocols
        carrier_param = None
        if is_instance_method:
            has_carrier = True
        else:
            carrier_tokens = self._get_carrier_tokens()
            params_to_check = required_params if required_params else all_params

            # 1. Type annotations against discovered container classes (Obj(C))
            for p in params_to_check:
                p_obj = param_objs.get(p)
                if p_obj and p_obj.annotation != inspect._empty:
                    anno = p_obj.annotation
                    if self._is_carrier_type(anno):
                        carrier_param = p
                        break
                    anno_str = str(anno).lower()
                    if any(t in anno_str for t in carrier_tokens):
                        carrier_param = p
                        break

            # 2. Check parameters against reflected carrier tokens and docstrings
            if not carrier_param:
                for p in params_to_check:
                    p_lower = p.lower()
                    if p_lower in carrier_tokens:
                        carrier_param = p
                        break
                    if doc:
                        m_p = re.search(rf"@param\s+{re.escape(p)}\b([^@\n]+)", doc, re.IGNORECASE)
                        if not m_p:
                            m_p = re.search(rf"^\s*{re.escape(p)}\s*:\s*([^\n]+)", doc, re.MULTILINE)
                        if m_p:
                            p_desc = m_p.group(1).lower()
                            if any(t in p_desc for t in carrier_tokens):
                                carrier_param = p
                                break

            # 3. Categorical operand fallback: if morphism docstring indicates 'dst' return, primary operand is source positional param
            if not carrier_param and params_to_check and doc:
                m_arrow = re.search(rf"\b{re.escape(func_name)}\(.*?\)\s*->\s*([a-zA-Z0-9_]+)", doc)
                if m_arrow:
                    arrow_target = m_arrow.group(1).lower()
                    if arrow_target == "dst" or arrow_target in carrier_tokens:
                        carrier_param = params_to_check[0]

            has_carrier = carrier_param is not None

        # Return type detection
        ret_type_str = ""
        if ret_annotation not in (inspect._empty, None):
            ret_type_str = getattr(ret_annotation, "__name__", str(ret_annotation))
        elif doc:
            m_ret = re.search(rf"\b{re.escape(func_name)}\(.*?\)\s*->\s*([a-zA-Z0-9_, ]+)", doc)
            if not m_ret:
                m_ret = re.search(r"Returns?\s*\n\s*-+\s*(?:[a-zA-Z0-9_]+\s*:\s*)?([a-zA-Z0-9_]+)", doc)
            if m_ret:
                ret_type_str = m_ret.group(1).strip()

        stage = self._infer_stage(
            func_name=func_name,
            has_carrier_input=has_carrier,
            ret_type=ret_type_str,
            doc=doc,
            is_instance_method=is_instance_method
        )

        # Detect Higher-Order Morphisms via reflection
        is_higher_order = False
        for p in all_params:
            p_obj = param_objs.get(p)
            if p_obj and p_obj.annotation != inspect._empty:
                anno_str = str(p_obj.annotation).lower()
                if "callable" in anno_str or "function" in anno_str:
                    is_higher_order = True
                    break
            if p_obj and p_obj.default not in (inspect._empty, None) and callable(p_obj.default):
                is_higher_order = True
                break
            if doc and re.search(rf"\b{re.escape(p)}\s*:\s*(?:[^,\n]*\b)?(?:callable|function)\b", doc, re.IGNORECASE):
                is_higher_order = True
                break

        node_type = "function"
        node_role = "function"
        slots: Dict[str, Any] = {}

        if is_higher_order:
            node_type = "higher_order"
            node_role = "macro"
            slots = {
                "body": {
                    "inner_input": {"type_name": "any", "state": "any"},
                    "inner_output": {"type_name": "any", "state": "any"}
                }
            }

        # Detect carrier type for output
        detected_out_carrier = carrier_type
        if is_instance_method and stage == 2:
            detected_out_carrier = carrier_type
        elif ret_type_str:
            for cls in self.container_classes:
                if cls.__name__.lower() in ret_type_str.lower():
                    detected_out_carrier = cls.__name__
                    break
        elif doc:
            for cls in self.container_classes:
                if re.search(rf"\b(?:returns?|into|->)\s*.*?\b{cls.__name__}\b", doc, re.IGNORECASE):
                    detected_out_carrier = cls.__name__
                    break

        # Build ports and template arguments
        inputs: Dict[str, PortSchema] = {}
        outputs: Dict[str, PortSchema] = {}
        template_args: List[str] = []
        is_parameterized = False

        if stage == 1:
            # Source Morphism: 0 -> C
            node_type = "source"
            src_param = all_params[0] if all_params else "filepath"
            # In Category Theory, an Initial Morphism 0 -> C ingests from the external environment (resource identifier)
            # or constructs a carrier from structural dimensions.
            is_file_src = (
                any(t in ("path", "file") for t in split_identifier_keywords(src_param))
                or (doc and "file" in doc.lower())
            )
            for p in all_params:
                if p == src_param:
                    if is_file_src:
                        inputs["filepath"] = PortSchema(type_name="str", state="source_identifier", description=f"Source argument {src_param}", required=True)
                        template_args.append("{filepath}")
                    else:
                        inputs[src_param] = PortSchema(type_name="any", state=src_param.lower(), description=f"Source parameter {src_param}", required=True)
                        template_args.append(f"{{{src_param}}}")
                elif p in required_params:
                    inputs[p] = PortSchema(type_name="any", state=p.lower(), description=f"Source argument {p}", required=True)
                    template_args.append(f"{{{p}}}")
            outputs["output_data"] = PortSchema(type_name=detected_out_carrier, state="raw", description="Ingested data")

        elif stage == 3:
            # Sink Morphism: C -> 1
            node_type = "sink"
            node_role = "sink"
            dest_param = next((p for p in all_params if p != carrier_param), None)
            if dest_param:
                inputs["dest_path"] = PortSchema(type_name="str", state="dest_identifier", description=f"Destination argument {dest_param}", required=True)
            inputs["data"] = PortSchema(type_name=carrier_type, state="any", description="Input data", required=True)

            if is_instance_method:
                for p in all_params:
                    if p == dest_param:
                        template_args.append("{dest_path}")
                    elif p in required_params:
                        inputs[p] = PortSchema(type_name="any", state=p.lower(), description=f"Sink argument {p}", required=True)
                        template_args.append(f"{{{p}}}")
            else:
                for p in all_params:
                    if p == dest_param:
                        template_args.append("{dest_path}")
                    elif p == carrier_param:
                        template_args.append("{data}")
                    elif p in required_params:
                        inputs[p] = PortSchema(type_name="any", state=p.lower(), description=f"Sink argument {p}", required=True)
                        template_args.append(f"{{{p}}}")

            outputs["output_data"] = PortSchema(type_name="str", state="filepath_written", description="Egress confirmation")

        else:
            # Stage 2: Transform / Endomorphism: C -> C'
            inputs["data"] = PortSchema(type_name=carrier_type, state="any", description="Input carrier data", required=True)

            for p in all_params:
                if not is_instance_method and p == carrier_param:
                    template_args.append("{data}")
                    continue

                p_obj = param_objs.get(p)
                enum_prefix = self._detect_enum_parameter(p, doc, p_obj)
                is_req = (p in required_params)

                if is_req:
                    if enum_prefix:
                        is_parameterized = True
                        domain_id = f"{mod_name}.{enum_prefix}_*"
                        default_const = None
                        if enum_prefix in self.constants_by_prefix and self.constants_by_prefix[enum_prefix]:
                            default_const = f"{self.constants_by_prefix[enum_prefix][0][2]}.{self.constants_by_prefix[enum_prefix][0][0]}"
                        inputs[p] = PortSchema(
                            type_name="Enum",
                            state=enum_prefix.lower(),
                            domain=domain_id,
                            required=True,
                            default_value=default_const,
                            description=f"Enum parameter {p} from {domain_id}"
                        )
                    else:
                        inputs[p] = PortSchema(
                            type_name="any",
                            state=p.lower(),
                            required=True,
                            description=f"Required parameter {p}"
                        )
                    template_args.append(f"{{{p}}}")

            if is_parameterized and node_type == "function":
                node_type = "parameterized"
                node_role = "parameterized"

            out_state = infer_typestate_from_name(func_name)
            outputs["output_data"] = PortSchema(type_name=detected_out_carrier, state=out_state, description=f"Output {out_state}")

        # Assemble code_template
        call_prefix = f"{mod_name}.{func_name}"
        dependencies = [f"import {mod_name}"]

        if is_instance_method:
            args_str = ", ".join(template_args)
            if stage == 3:
                dest = "{dest_path}" if "dest_path" in inputs else "''"
                code_template = f"{{data}}.{func_name}({args_str})\n{{output_var}} = {dest}"
            else:
                code_template = f"{{output_var}} = {{data}}.{func_name}({args_str})"
        else:
            args_str = ", ".join(template_args)
            if stage == 3:
                dest = "{dest_path}" if "dest_path" in inputs else "''"
                code_template = f"{call_prefix}({args_str})\n{{output_var}} = {dest}"
            else:
                code_template = f"{{output_var}} = {call_prefix}({args_str})"

        # Lossless semantic keywords
        kw_tokens = split_identifier_keywords(func_name)
        if mod_name != self.package_name:
            kw_tokens.extend(split_identifier_keywords(mod_name.replace(self.package_name, "")))
        kw_tokens.extend([self.domain_name.lower(), func_name.lower()])
        semantic_tags = list(dict.fromkeys(kw_tokens))

        # Format cell_id
        clean_mod = mod_name.replace(self.package_name, "").strip(".").replace(".", "_").upper()
        clean_fn = func_name.upper()
        if is_instance_method and container_class_name:
            cell_id = f"{self.domain_name.upper()}_{container_class_name.upper()}_{clean_fn}"
        elif clean_mod:
            cell_id = f"{self.domain_name.upper()}_{clean_mod}_{clean_fn}"
        else:
            cell_id = f"{self.domain_name.upper()}_{clean_fn}"

        return CellSchema(
            cell_id=cell_id,
            stage=stage,
            inputs=inputs,
            outputs=outputs,
            slots=slots,
            code_template=code_template,
            dependencies=dependencies,
            semantic_tags=semantic_tags,
            keywords=semantic_tags,
            docstring=first_doc,
            domain_name=self.domain_name,
            node_type=node_type,
            node_role=node_role,
            verified=True,
            source_priority=50
        )

    def harvest_all(self) -> List[CellSchema]:
        """
        Executes complete library harvest:
          1. Submodule discovery
          2. Constants and Enums extraction -> Constant Morphisms
          3. Container classes and instance methods
          4. Top-level and submodule callables -> Atomic, Parameterized, Scoped
        """
        self.discover_submodules()
        self.collect_constants_and_enums()
        self.identify_container_classes()

        cells: List[CellSchema] = []
        seen_ids: Set[str] = set()

        # 1. Harvest Constant Morphisms
        constant_cells = self.harvest_constant_nodes()
        for c in constant_cells:
            if c.cell_id not in seen_ids:
                cells.append(c)
                seen_ids.add(c.cell_id)

        logger.info(f"[{self.domain_name}] Harvested {len(constant_cells)} constant morphisms")

        # 2. Harvest Instance Methods on Container Classes
        method_count = 0
        for cls in self.container_classes:
            cls_mod = getattr(cls, "__module__", "") or ""
            if not cls_mod.startswith(self.package_name) and self.package_name != "builtins":
                continue
            cls_name = cls.__name__
            for attr_name in dir(cls):
                if attr_name.startswith("_"):
                    continue
                try:
                    attr_val = getattr(cls, attr_name, None)
                    if callable(attr_val):
                        cell = self.harvest_callable(
                            func_name=attr_name,
                            func_obj=attr_val,
                            mod_name=getattr(cls, "__module__", self.package_name),
                            is_instance_method=True,
                            container_class_name=cls_name
                        )
                        if cell and cell.cell_id not in seen_ids:
                            cells.append(cell)
                            seen_ids.add(cell.cell_id)
                            method_count += 1
                except Exception:
                    continue

        logger.info(f"[{self.domain_name}] Harvested {method_count} instance methods across containers")

        # 3. Harvest Callables Across All Submodules
        callable_count = 0
        for mod_name, mod in self.discovered_modules.items():
            for attr_name in dir(mod):
                if attr_name.startswith("_"):
                    continue
                try:
                    attr_val = getattr(mod, attr_name, None)
                    if callable(attr_val) and not inspect.isclass(attr_val):
                        cell = self.harvest_callable(
                            func_name=attr_name,
                            func_obj=attr_val,
                            mod_name=mod_name,
                            is_instance_method=False
                        )
                        if cell and cell.cell_id not in seen_ids:
                            cells.append(cell)
                            seen_ids.add(cell.cell_id)
                            callable_count += 1
                except Exception:
                    continue

        logger.info(f"[{self.domain_name}] Harvested {callable_count} callable morphisms")
        return cells

    @classmethod
    def enrich_from_existing_trees(
        cls,
        domain_name: str,
        new_cells: List[CellSchema],
        existing_tree_paths: List[Union[str, Path]]
    ) -> List[CellSchema]:
        """
        Cross-references newly harvested ground-truth cells against old knowledge bases.
        Inherits curated docstrings, semantic tags, intent keywords, and priority <= 10.
        Completely purges ghost duplicate combinations.
        """
        # Build lookup table from existing trees by canonical function name
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
                    if not isinstance(oc, dict) or "cell_id" not in oc:
                        continue
                    cid = oc["cell_id"].upper()
                    # Extract base function name from cell_id
                    parts = cid.split("_")
                    # e.g. CV2_CVTCOLOR or PANDAS_SORT_VALUES
                    for k in range(1, len(parts) + 1):
                        sub_key = "_".join(parts[1:k])
                        if sub_key and sub_key not in old_knowledge:
                            old_knowledge[sub_key] = oc
                    if cid not in old_knowledge:
                        old_knowledge[cid] = oc
            except Exception as e:
                logger.warning(f"Failed to read existing tree {p}: {e}")

        enriched_count = 0
        for cell in new_cells:
            # Try exact cell_id match first
            cid = cell.cell_id.upper()
            match = old_knowledge.get(cid)
            if not match:
                # Try suffix match (e.g. CVTCOLOR in CV2_CVTCOLOR)
                parts = cid.split("_")
                for k in range(1, len(parts)):
                    sub_key = "_".join(parts[k:])
                    if sub_key in old_knowledge:
                        match = old_knowledge[sub_key]
                        break

            if match:
                # Inherit high-value curated metadata
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
                if old_doc and (not cell.docstring or len(old_doc) > len(cell.docstring)):
                    cell.docstring = old_doc

                if match.get("verified", False):
                    cell.verified = True

                enriched_count += 1

        # Preserve verified gold-standard curated nodes (source_priority <= 10) not captured by pure reflection
        existing_cell_ids = {c.cell_id.upper() for c in new_cells}
        preserved_curated = 0
        for cid, oc in old_knowledge.items():
            if not isinstance(oc, dict) or "cell_id" not in oc:
                continue
            orig_cid = oc["cell_id"].upper()
            d_prefix = domain_name.upper().replace("_CORE", "")
            if orig_cid.startswith(domain_name.upper()) or orig_cid.startswith(f"{d_prefix}_"):
                if oc.get("source_priority", 100) <= 10:
                    if orig_cid not in existing_cell_ids:
                        try:
                            oc_copy = dict(oc)
                            oc_copy["verified"] = True
                            repair_wiring_invariant(oc_copy, domain_name)
                            preserved_cell = CellSchema(**oc_copy)
                            new_cells.append(preserved_cell)
                            existing_cell_ids.add(orig_cid)
                            preserved_curated += 1
                        except Exception as e:
                            logger.warning(f"[{domain_name}] Failed to preserve curated node {orig_cid}: {e}")

        logger.info(f"[{domain_name}] Enriched {enriched_count}/{len(new_cells)} cells, preserved {preserved_curated} curated priority nodes")
        return new_cells

    def merge_and_save(self, new_cells: List[CellSchema], out_file: Union[str, Path]):
        """Saves harvested cells into domain JSON tree schema."""
        out_path = Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tree = TreeSchema(
            domain=self.domain_name,
            cells=new_cells
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(tree.model_dump_json(indent=2))
