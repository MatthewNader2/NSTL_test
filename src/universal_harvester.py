"""
src/universal_harvester.py - Neuro-Symbolic Topological Lattice (NSTL)
Universal Dynamic Library Harvester & Morphism Generator.

Domain-agnostic runtime introspection engine that can ingest any arbitrary Python package
purely at runtime and generate clean, mathematically categorized lattice nodes conforming
to category-theoretic archetypes:
  1. Constant Morphisms (* -> Enum / Const, Stage 0)
  2. Initial / Source Ingestion Morphisms (0 -> A, Stage 1)
  3. Endomorphism / Transform Morphisms (A -> A, Stage 2)
  4. Bridge / Tunnel Morphisms (A -> B, A != B, Stage 2)
  5. Terminal / Sink Egress Morphisms (A -> 1, Stage 3)

ZERO hardcoded library lists, ZERO hardcoded primitive sets, ZERO regex.
"""

from __future__ import annotations
import importlib
import inspect
import json
import pkgutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from schema import CellSchema, PortSchema, TreeSchema
    from log_config import get_logger
    from tokenizer import CellTokenizer
    from signature_introspector import get_callable_parameters
except ImportError:
    from .schema import CellSchema, PortSchema, TreeSchema
    from .log_config import get_logger
    from .tokenizer import CellTokenizer
    from .signature_introspector import get_callable_parameters

logger = get_logger("universal_harvester")


def extract_type_name(anno: Any) -> str:
    """
    Extracts canonical type name from any annotation or type object dynamically.
    Pure runtime reflection with zero hardcoded type lists.
    """
    if anno is None or anno is inspect.Signature.empty or anno is inspect.Parameter.empty:
        return "any"

    # Runtime class objects (e.g. class types)
    if inspect.isclass(anno):
        return anno.__name__

    # Generics and Union/Optional types
    origin = getattr(anno, "__origin__", None)
    args = getattr(anno, "__args__", None)
    if origin is not None and args:
        # For Union/Optional, pick the first non-None alternative
        non_none = [a for a in args if getattr(a, "__name__", "") not in ("NoneType", "None")]
        if non_none:
            return extract_type_name(non_none[0])
        return "None"

    # Objects declaring __name__
    name = getattr(anno, "__name__", None)
    if name:
        return str(name)

    # String representation from docstrings or forward references
    s = str(anno).strip()
    if not s:
        return "any"

    # Strip trailing descriptions outside brackets/parens: e.g. "int, optional"
    if "," in s and "[" not in s and "(" not in s:
        s = s.split(",")[0].strip()

    # Strip parenthesized shape/dimension prefixes: e.g. "(..., M, M) array_like"
    if s.startswith("(") and ")" in s:
        s = s[s.find(")") + 1 :].strip()

    # Filter tensor rank dimension prefixes: e.g. "1D or 2D array_like"
    words = s.split()
    filtered = []
    for w in words:
        w_clean = w.rstrip(",")
        if (w_clean.endswith(("D", "d")) and w_clean[:-1].isdigit()) or w_clean.lower() == "or":
            continue
        filtered.append(w)
    if filtered:
        s = " ".join(filtered)

    # Decompose coproducts/unions first (| and or)
    if "|" in s:
        parts = [p.strip() for p in s.split("|") if p.strip().lower() not in ("none", "nonetype")]
        if parts:
            s = parts[0]
        else:
            return "None"

    if " or " in s:
        parts = [p.strip() for p in s.split(" or ") if p.strip().lower() not in ("none", "nonetype")]
        if parts:
            s = parts[0]
        else:
            return "None"

    # Bracketed generic wrappers: Optional[X] / Union[X] -> X, List[X] -> List
    if s.startswith(("Optional[", "Union[")) and "]" in s:
        s = s[s.find("[") + 1 : s.rfind("]")].strip()
        parts = [p.strip() for p in s.split(",") if p.strip().lower() not in ("none", "nonetype")]
        if parts:
            s = parts[0]
    elif "[" in s and "]" in s:
        s = s[: s.find("[")].strip()

    # Qualified module path: e.g. foo.bar.Baz -> Baz
    if "." in s:
        s = s.split(".")[-1].strip()

    # Final identifier token
    words = s.split()
    if words:
        s = words[0]

    s = s.rstrip(".,;:)>]`\"'")
    if s.lower() in ("retval", "result", "res", "return", "value"):
        return "any"
    return s if s else "any"


def extract_doc_types(doc: str) -> Dict[str, str]:
    """
    Parses parameter and return types from docstrings across Sphinx, NumPy, Google, and Doxygen conventions.
    Pure deterministic line-by-line scanner with zero regular expressions.
    Returns: {"__return__": <return_type>, "__return_raw__": <raw_str>, <param_name>: <param_type>, ...}
    """
    if not doc:
        return {}

    types: Dict[str, str] = {}
    lines = doc.splitlines()
    n = len(lines)
    i = 0

    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        clean_dox = stripped.lstrip(". \t")

        # 1. Sphinx format: :type <param>: <type>
        if stripped.startswith(":type "):
            rest = stripped[6:].strip()
            if ":" in rest:
                p_name, p_type = rest.split(":", 1)
                types[p_name.strip()] = extract_type_name(p_type.strip())
                types["__has_params_doc__"] = "true"
            i += 1
            continue

        # Sphinx return: :rtype: <type> or :return: / :returns:
        if stripped.startswith((":rtype:", ":return:", ":returns:")):
            val = stripped.split(":", 2)[-1].strip()
            if val:
                types["__return_raw__"] = val
                types["__return__"] = extract_type_name(val)
                types["__has_returns_doc__"] = "true"
            i += 1
            continue

        # 2. NumPy format: Section header followed by dashed underline
        if i + 1 < n:
            next_stripped = lines[i + 1].strip()
            if next_stripped and len(next_stripped) >= 3 and set(next_stripped) == {"-"}:
                sec = stripped.lower()
                i += 2
                if sec in ("returns", "yields"):
                    types["__has_returns_doc__"] = "true"
                    while i < n:
                        curr = lines[i].strip()
                        if not curr:
                            i += 1
                            continue
                        if i + 1 < n and set(lines[i + 1].strip()) == {"-"}:
                            break
                        if " : " in curr:
                            raw = curr.split(":", 1)[1].strip().split(",")[0].strip()
                        else:
                            raw = curr.split(",")[0].strip()
                        types["__return_raw__"] = raw
                        types["__return__"] = extract_type_name(raw)
                        break
                    continue
                elif sec in ("parameters", "other parameters", "args", "arguments"):
                    types["__has_params_doc__"] = "true"
                    while i < n:
                        curr = lines[i].strip()
                        if not curr:
                            i += 1
                            continue
                        if i + 1 < n and set(lines[i + 1].strip()) == {"-"}:
                            break
                        if not lines[i].startswith(" ") and not lines[i].startswith("\t"):
                            if " : " in curr:
                                p_name, p_type = curr.split(":", 1)
                                types[p_name.strip()] = extract_type_name(p_type.strip())
                        i += 1
                    continue

        # 3. Google format: Args: / Returns:
        if stripped == "Args:":
            types["__has_params_doc__"] = "true"
            i += 1
            while i < n and (lines[i].startswith("    ") or lines[i].startswith("\t") or not lines[i].strip()):
                curr = lines[i].strip()
                if curr and "(" in curr and ")" in curr and ":" in curr:
                    left = curr.split(":", 1)[0]
                    p_name = left.split("(")[0].strip()
                    p_type = left.split("(")[1].split(")")[0].strip()
                    if p_name and p_type:
                        types[p_name] = extract_type_name(p_type)
                i += 1
            continue

        if stripped == "Returns:":
            types["__has_returns_doc__"] = "true"
            i += 1
            while i < n and not lines[i].strip():
                i += 1
            if i < n and (lines[i].startswith("    ") or lines[i].startswith("\t")):
                curr = lines[i].strip()
                if ":" in curr:
                    raw = curr.split(":", 1)[0].strip()
                else:
                    raw = curr.split()[0].strip()
                types["__return_raw__"] = raw
                types["__return__"] = extract_type_name(raw)
            continue

        # 4. Doxygen format: @param <name> [<type>] or @return <type>
        if clean_dox.startswith("@param "):
            types["__has_params_doc__"] = "true"
            rest = clean_dox[7:].strip()
            words = rest.split()
            if words:
                p_name = words[0]
                if "(" in rest and ")" in rest:
                    inside = rest[rest.find("(") + 1 : rest.find(")")].strip()
                    types[p_name] = extract_type_name(inside)
                elif len(words) > 1 and not words[1].startswith("@"):
                    types[p_name] = extract_type_name(words[1])
            i += 1
            continue

        if clean_dox.startswith(("@return ", "@returns ")):
            types["__has_returns_doc__"] = "true"
            val = clean_dox.split(None, 1)[-1].strip()
            if val and "__return__" not in types:
                types["__return_raw__"] = val
                types["__return__"] = extract_type_name(val)
            i += 1
            continue

        # 5. Signature line arrow: func(...) -> Type
        if "->" in stripped and ("(" in stripped or stripped.startswith("->")):
            parts = stripped.split("->", 1)
            raw = parts[1].strip().split()[0].rstrip(".,;:)")
            if raw and "__return__" not in types:
                types["__has_returns_doc__"] = "true"
                types["__return_raw__"] = raw
                types["__return__"] = extract_type_name(raw)

        i += 1

    return types


class UniversalHarvester:
    """
    Truly universal, domain-agnostic library introspection engine.
    Ingests any arbitrary Python package and extracts typed category-theoretic nodes.
    """

    def __init__(self, domain_name: str, package_name: Optional[str] = None):
        self.domain_name = domain_name
        self.package_name = package_name or domain_name

        # Dynamic package import
        try:
            self.root_module = importlib.import_module(self.package_name)
        except ImportError:
            if str(Path.cwd()) not in sys.path:
                sys.path.insert(0, str(Path.cwd()))
            self.root_module = importlib.import_module(self.package_name)

        self.discovered_modules: Dict[str, Any] = {}

    def discover_modules(self) -> Dict[str, Any]:
        """Discovers all public submodules in the package via pkgutil."""
        modules = {self.package_name: self.root_module}
        pkg_path = getattr(self.root_module, "__path__", None)

        def _walk(path, prefix):
            try:
                for item in pkgutil.iter_modules(path, prefix):
                    parts = item.name.split(".")
                    # Skip private modules (unless root C-extension _pkg) and test directories
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

    def harvest_all(self) -> List[CellSchema]:
        """
        Executes complete library harvest:
          1. Submodule discovery & domain class collection (Obj(C))
          2. Uppercase constants -> Constant Morphisms (* -> Enum / Const, Stage 0)
          3. Public classes and instance methods -> Endomorphisms & Bridges (Stage 2) / Sinks (Stage 3)
          4. Module-level callables -> Sources (Stage 1), Transforms / Bridges (Stage 2), Sinks (Stage 3)
        """
        if not self.discovered_modules:
            self.discover_modules()

        cells: List[CellSchema] = []
        seen_ids: Set[str] = set()

        # Collect all domain carrier classes defined within the package (Obj(C))
        package_classes: Set[str] = set()
        for m_name, m in self.discovered_modules.items():
            for c_name in dir(m):
                if c_name.startswith("_"):
                    continue
                try:
                    cls = getattr(m, c_name, None)
                    if inspect.isclass(cls):
                        cls_mod = getattr(cls, "__module__", "") or ""
                        if cls_mod.startswith(self.package_name) or cls_mod.startswith(f"_{self.package_name}"):
                            package_classes.add(c_name)
                except Exception:
                    continue

        # 1. Harvest Constant Morphisms (* -> Const, Stage 0)
        for m_name, m in self.discovered_modules.items():
            for attr_name in dir(m):
                if not attr_name.isupper() or attr_name.startswith("_"):
                    continue
                try:
                    val = getattr(m, attr_name, None)
                    if val is None or callable(val):
                        continue

                    cid = f"{self.domain_name.upper()}_{attr_name}"
                    if cid in seen_ids:
                        continue
                    seen_ids.add(cid)

                    val_type = type(val).__name__
                    tokens = sorted(list(CellTokenizer.tokenize_identifier(attr_name)))
                    cell = CellSchema(
                        cell_id=cid,
                        stage=0,
                        inputs={},
                        outputs={"value": PortSchema(type_name=val_type, state="constant", domain=f"{m_name}.{attr_name}")},
                        code_template=f"{m_name}.{attr_name}",
                        dependencies=[f"import {m_name}"],
                        semantic_tags=tokens,
                        keywords=tokens,
                        docstring=f"Constant {m_name}.{attr_name}",
                        domain_name=self.domain_name,
                        node_type="constant",
                        node_role="constant",
                        source_priority=100
                    )
                    cells.append(cell)
                except Exception:
                    continue

        # 2. Harvest Classes and Methods
        for m_name, m in self.discovered_modules.items():
            for c_name in dir(m):
                if c_name.startswith("_"):
                    continue
                try:
                    cls = getattr(m, c_name, None)
                    if not inspect.isclass(cls):
                        continue
                    cls_mod = getattr(cls, "__module__", "") or ""
                    if not (cls_mod.startswith(self.package_name) or cls_mod.startswith(f"_{self.package_name}")):
                        continue

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
                            seen_ids.add(cid)

                            doc = inspect.getdoc(fn) or ""
                            first_doc = doc.splitlines()[0] if doc else f"{c_name}.{m_name_attr}"
                            doc_types = extract_doc_types(doc)

                            # Introspect method signature
                            sig = None
                            try:
                                sig = inspect.signature(fn)
                            except Exception:
                                pass

                            params = list(sig.parameters.values()) if sig else []
                            # Strip receiver (self / cls)
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
                            template_args: List[str] = []

                            for p in params:
                                p_type = extract_type_name(p.annotation)
                                if p_type == "any" and p.name in doc_types:
                                    p_type = doc_types[p.name]
                                if p_type == "any" and p.default is not inspect.Parameter.empty:
                                    p_type = type(p.default).__name__

                                is_req = (p.default is inspect.Parameter.empty)
                                inputs[p.name] = PortSchema(
                                    type_name=p_type,
                                    state=p.name.lower(),
                                    required=is_req,
                                    description=f"Argument {p.name}"
                                )
                                template_args.append(f"{{{p.name}}}")

                            # Return type
                            ret_type = "any"
                            if sig and sig.return_annotation is not inspect.Signature.empty:
                                ret_type = extract_type_name(sig.return_annotation)
                            if ret_type == "any" and "__return__" in doc_types:
                                ret_type = doc_types["__return__"]

                            # Detect destination writing parameter & optional return
                            param_names = [p.name for p in params]
                            has_dest_param = any(any(d in p_n.lower() for d in ("dest", "path", "file", "fname", "buf", "target")) for p_n in param_names)
                            doc_includes_none = ("none" in doc_types.get("__return_raw__", "").lower())

                            # Morphism Archetype Categorization
                            if ret_type in ("None", "NoneType", "void", "NoReturn") or (has_dest_param and doc_includes_none):
                                stage = 3
                                node_type = "sink"
                                node_role = "sink"
                                state = "destination_written"
                                out_type = "str"
                            elif ret_type != "any" and ret_type != c_name:
                                stage = 2
                                node_type = "bridge"
                                node_role = "tunnel"
                                state = "raw"
                                out_type = ret_type
                            else:
                                stage = 2
                                node_type = "function"
                                node_role = "transform"
                                state = "transformed"
                                out_type = c_name

                            args_str = ", ".join(template_args)
                            if stage == 3:
                                code_template = f"{{data}}.{m_name_attr}({args_str})\n{{output_var}} = 'done'"
                            else:
                                code_template = f"{{output_var}} = {{data}}.{m_name_attr}({args_str})"

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
                    fn = getattr(m, attr_name, None)
                    if not callable(fn) or inspect.isclass(fn):
                        continue
                    fn_mod = getattr(fn, "__module__", None) or m_name
                    if not (fn_mod.startswith(self.package_name) or fn_mod.startswith(f"_{self.package_name}")):
                        continue

                    cid = f"{self.domain_name.upper()}_{attr_name.upper()}"
                    if cid in seen_ids:
                        continue
                    seen_ids.add(cid)

                    doc = inspect.getdoc(fn) or ""
                    first_doc = doc.splitlines()[0] if doc else f"{m_name}.{attr_name}"
                    doc_types = extract_doc_types(doc)

                    sig = None
                    try:
                        sig = inspect.signature(fn)
                    except Exception:
                        pass

                    param_items: List[Tuple[str, Any, bool, Any]] = []
                    if sig:
                        for p in sig.parameters.values():
                            is_req = (p.default is inspect.Parameter.empty)
                            param_items.append((p.name, p.annotation, is_req, p.default))
                    else:
                        try:
                            c_info = get_callable_parameters(fn, attr_name)
                            if c_info:
                                req_set = set(c_info.get("required", []))
                                for p_n in c_info.get("all", []):
                                    param_items.append((p_n, inspect.Parameter.empty, p_n in req_set, inspect.Parameter.empty))
                        except Exception:
                            pass

                    inputs: Dict[str, PortSchema] = {}
                    template_args: List[str] = []
                    param_names: List[str] = []
                    input_types: List[str] = []

                    for idx, (p_name, p_anno, is_req, p_default) in enumerate(param_items):
                        p_type = extract_type_name(p_anno)
                        if p_type == "any" and p_name in doc_types:
                            p_type = doc_types[p_name]
                        if p_type == "any" and p_default is not inspect.Parameter.empty:
                            p_type = type(p_default).__name__

                        inputs[p_name] = PortSchema(
                            type_name=p_type,
                            state=p_name.lower(),
                            required=is_req,
                            description=f"Argument {p_name}"
                        )
                        template_args.append(f"{{{p_name}}}")
                        param_names.append(p_name)
                        input_types.append(p_type)

                    primary_in_type = input_types[0] if input_types else "any"

                    ret_type = "any"
                    if sig and sig.return_annotation is not inspect.Signature.empty:
                        ret_type = extract_type_name(sig.return_annotation)
                    if ret_type == "any" and "__return__" in doc_types:
                        ret_type = doc_types["__return__"]
                    if ret_type == "any" and doc_types.get("__has_params_doc__") and not doc_types.get("__has_returns_doc__"):
                        ret_type = "None"

                    consumes_domain = any(t in package_classes for t in input_types)
                    produces_domain = (ret_type in package_classes)
                    has_source_param = any(any(s in p_n.lower() for s in ("path", "file", "fname", "source", "url", "buf")) for p_n in param_names)
                    has_dest_param = any(any(d in p_n.lower() for d in ("dest", "path", "file", "fname", "buf", "target")) for p_n in param_names)
                    doc_includes_none = ("none" in doc_types.get("__return_raw__", "").lower())

                    # Category-Theoretic Morphism Archetype Categorization
                    if ret_type in ("None", "NoneType", "void", "NoReturn") or (has_dest_param and consumes_domain) or (has_dest_param and (doc_includes_none or not doc_types.get("__has_returns_doc__"))):
                        stage = 3
                        node_type = "sink"
                        node_role = "sink"
                        state = "destination_written"
                        out_type = "str"
                    elif not consumes_domain and (len(param_items) == 0 or has_source_param or produces_domain):
                        stage = 1
                        node_type = "source"
                        node_role = "source"
                        state = "raw"
                        if ret_type != "any":
                            out_type = ret_type
                        else:
                            matched_cls = next((c for c in package_classes if c.lower() in doc.lower()), None)
                            out_type = matched_cls if matched_cls else (next(iter(package_classes)) if package_classes else "any")
                    elif primary_in_type != "any" and ret_type != "any" and primary_in_type != ret_type:
                        stage = 2
                        node_type = "bridge"
                        node_role = "tunnel"
                        state = "raw"
                        out_type = ret_type
                    else:
                        stage = 2
                        node_type = "function"
                        node_role = "transform"
                        state = "transformed"
                        out_type = ret_type

                    args_str = ", ".join(template_args)
                    if stage == 3:
                        code_template = f"{m_name}.{attr_name}({args_str})\n{{output_var}} = 'done'"
                    else:
                        code_template = f"{{output_var}} = {m_name}.{attr_name}({args_str})"

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
        """
        Cross-references newly harvested ground-truth cells against old knowledge bases.
        Inherits curated docstrings, semantic tags, intent keywords, and priority <= 10.
        Never overwrites extracted input/output types or bridge roles.
        """
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
        """Harvests library and saves tree JSON."""
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

    # Print summary of archetypes
    by_archetype: Dict[str, int] = {}
    bridges: List[CellSchema] = []
    for c in tree.cells:
        k = f"Stage {c.stage} | Type: {c.node_type} | Role: {c.node_role}"
        by_archetype[k] = by_archetype.get(k, 0) + 1
        if c.node_role == "tunnel" or c.node_type == "bridge":
            bridges.append(c)

    print(f"\n[NSTL Universal Harvester] Completed harvest of '{lib}':")
    print(f"Total cells generated: {len(tree.cells)}")
    print("\nArchetype breakdown:")
    for k, count in sorted(by_archetype.items()):
        print(f"  {k}: {count} nodes")

    if bridges:
        print(f"\nSample Bridge / Tunnel nodes ({len(bridges)} total):")
        for b in bridges[:5]:
            in_types = [(k, v.type_name, v.state) for k, v in b.inputs.items()]
            out_types = [(k, v.type_name, v.state) for k, v in b.outputs.items()]
            print(f"  {b.cell_id}: {in_types} -> {out_types}")
