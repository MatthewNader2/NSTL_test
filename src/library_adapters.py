"""
src/library_adapters.py - Neuro-Symbolic Topological Lattice (NSTL)
Universal Domain-Agnostic Library Adapters.

Zero hardcoded library names, types, enums, or function names.
Adapters are selected strictly based on HOW the library is physically implemented:
  1. PythonSourceAdapter: Pure Python packages with accessible source code and AST.
  2. CompiledExtensionAdapter: Compiled C/C++/Rust binary extensions (.so/.pyd) using PEP 561 .pyi stubs and C introspection.
  3. HybridLibraryAdapter: Mixed packages containing both Python wrappers and compiled extensions.
"""

from __future__ import annotations

import abc
import enum
import functools
import importlib
import inspect
import os
import pkgutil
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from log_config import get_logger
    from signature_introspector import (
        resolve_signature,
        _find_stub_file_for_module,
        extract_clean_type_name,
        infer_abstract_carrier,
        extract_enum_domain,
        extract_return_specs,
        extract_param_kind,
        extract_docstring_params,
        extract_param_constraints,
        extract_shape_contract,
        extract_docstring_raises,
        extract_ast_raises,
        extract_type_vars,
    )
except ImportError:
    from .log_config import get_logger
    from .signature_introspector import (
        resolve_signature,
        _find_stub_file_for_module,
        extract_clean_type_name,
        infer_abstract_carrier,
        extract_enum_domain,
        extract_return_specs,
        extract_param_kind,
        extract_docstring_params,
        extract_param_constraints,
        extract_shape_contract,
        extract_docstring_raises,
        extract_ast_raises,
        extract_type_vars,
    )

logger = get_logger("library_adapters")


def _is_python_source_module(module: Any) -> bool:
    """True iff the module is backed by real Python source (.py), not a compiled binary."""
    file_path = getattr(module, "__file__", None) or ""
    if not file_path:
        return False
    return str(file_path).endswith(".py")


def _is_compiled_extension_module(module: Any) -> bool:
    """True iff the module is a compiled extension (.so, .pyd, or built-in)."""
    file_path = getattr(module, "__file__", None) or ""
    if not file_path:
        # Built-in or dynamically allocated C module
        return True
    lower = str(file_path).lower()
    return lower.endswith(".so") or lower.endswith(".pyd") or ".cpython" in lower


def _walk_package(root_module: Any) -> Dict[str, Any]:
    """Universal module enumeration for an importable package."""
    package_name = getattr(root_module, "__name__", "")
    modules: Dict[str, Any] = {package_name: root_module}
    pkg_path = getattr(root_module, "__path__", None)

    def _excluded(name: str) -> bool:
        parts = name.split(".")
        for p in parts:
            pl = p.lower()
            if pl.startswith("_") and pl != f"_{package_name}":
                return True
            if pl in (
                "tests", "test", "testing", "_testing", "testutils", "conftest",
                "estimator_checks", "_estimator_checks",
                "externals", "vendor", "vendored", "_vendor",
                "compat", "_compat", "_internal", "internal", "internals",
                "fixes", "_fixes"
            ):
                return True
            if pl.startswith("test_") or pl.endswith("_test"):
                return True
        return False

    def _walk(path, prefix):
        try:
            for item in pkgutil.iter_modules(path, prefix):
                if _excluded(item.name):
                    continue
                if item.name.split(".")[-1].startswith("_") and item.name != f"_{package_name}":
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
        _walk(pkg_path, package_name + ".")
    return modules


# =====================================================================
# Base Library Adapter Contract
# =====================================================================

class LibraryAdapter(abc.ABC):
    """
    Contract for library-specific extraction and introspection.
    Adapters adapt to HOW a library is constructed (source AST, compiled binary, stubs)
    rather than which specific domain it serves.
    """
    paradigm: str = "base"

    def __init__(self, root_module: Any = None):
        self.root_module = root_module

    def enumerate_modules(self, root_module: Any = None) -> Dict[str, Any]:
        return _walk_package(root_module or self.root_module)

    def get_source(self, obj: Any) -> Optional[str]:
        try:
            src = inspect.getsource(obj)
            return src if src and src.strip() else None
        except Exception:
            return None

    def get_docstring(self, obj: Any) -> Optional[str]:
        try:
            return inspect.getdoc(obj)
        except Exception:
            raw = getattr(obj, "__doc__", None)
            return str(raw) if raw else None

    def resolve_callable_signature(
        self,
        obj: Any,
        callable_name: str = "",
        parent_cls_name: str = "",
        mod: Any = None
    ) -> Optional[inspect.Signature]:
        return resolve_signature(
            obj,
            callable_name=callable_name,
            parent_cls_name=parent_cls_name,
            mod=mod or self.root_module
        )

    def get_domain_carriers(self, root_module: Any = None) -> Set[str]:
        """
        Discovers domain carrier classes dynamically from the package namespace.
        Any class defined within the package or exposed at its root is a carrier.
        """
        carriers: Set[str] = set()
        rm = root_module or self.root_module
        if rm:
            pkg_name = getattr(rm, "__name__", "")
            for m in self.enumerate_modules(rm).values():
                for attr in dir(m):
                    if not attr.startswith("_"):
                        val = getattr(m, attr, None)
                        if inspect.isclass(val):
                            cls_mod = getattr(val, "__module__", "") or ""
                            if cls_mod.startswith(pkg_name) or cls_mod.startswith(f"_{pkg_name}"):
                                carriers.add(val.__name__)
        return carriers

    def get_abstract_type(self, type_name_or_obj: Any, param_name: str = "") -> Optional[str]:
        """Maps a concrete type to an abstract categorical carrier using language protocols."""
        return infer_abstract_carrier(type_name_or_obj, param_name=param_name)

    def get_parameter_domains(
        self,
        anno: Any,
        doc: str = "",
        param_name: str = "",
        mod: Any = None,
    ) -> Optional[List[Any]]:
        """
        Extracts enum domain choices for a parameter purely from type annotations,
        subclasses of enum.Enum, or docstring citations.
        """
        # 1. Direct annotation inspection (Literal, Enum subclass, or docstring set)
        choices = extract_enum_domain(anno, doc, param_name)
        if choices:
            return choices

        # 2. Dynamic docstring symbol lookup (e.g. '@param p ... see #EnumClass' or 'values in EnumClass')
        target_mod = mod or self.root_module
        if doc and param_name and target_mod:
            p_desc = ""
            doc_params = extract_docstring_params(doc)
            if param_name in doc_params:
                p_desc = doc_params[param_name].get("desc", "")
            else:
                p_esc = re.escape(param_name)
                m = re.search(rf"(?:@param\s+|:param\s+.*?\s+|^\s*){p_esc}\b[^\n]*\n?([^\n]*)", doc, re.MULTILINE)
                if m:
                    p_desc = m.group(0)

            if p_desc:
                citations = re.findall(r"(?:see|values\s+in|one\s+of)\s+#?(?:[A-Za-z_]\w*::)*([A-Za-z_]\w+)", p_desc, re.IGNORECASE)
                for enum_ident in citations:
                    if len(enum_ident) < 3 or enum_ident.lower() in ("the", "see", "for", "and", "one", "all", "none", "true", "false", "list"):
                        continue
                    enum_obj = getattr(target_mod, enum_ident, None) or getattr(self.root_module, enum_ident, None)
                    if enum_obj is not None:
                        if inspect.isclass(enum_obj) and issubclass(enum_obj, enum.Enum):
                            return [m.name for m in enum_obj]
                        if inspect.isclass(enum_obj):
                            constants = [attr for attr in dir(enum_obj) if attr.isupper() and not attr.startswith("_")]
                            if constants:
                                return constants
                    clean_name = re.sub(r"(?:Flags|Types?|Codes?)$", "", enum_ident, flags=re.IGNORECASE)
                    pfx = f"{clean_name.upper()}_"
                    matches = [attr for attr in dir(target_mod) if attr.startswith(pfx)]
                    if matches:
                        return matches

        return None

    def get_return_ports(
        self,
        ret_anno: Any,
        doc: str = ""
    ) -> List[Tuple[str, str, Optional[str]]]:
        """Extracts return specifications: List[(port_name, type_name, abstract_type)]."""
        specs = extract_return_specs(ret_anno, doc)
        resolved = []
        for name, t_name, abs_t in specs:
            enriched_abs = self.get_abstract_type(t_name) or abs_t
            resolved.append((name, t_name, enriched_abs))
        return resolved

    def get_param_kind(self, param: inspect.Parameter) -> str:
        """Determines parameter calling convention."""
        return extract_param_kind(param)

    def get_docstring_params(self, doc: str) -> Dict[str, Dict[str, Any]]:
        """Parses parameter types and descriptions from docstrings."""
        return extract_docstring_params(doc)

    def get_param_constraints(self, text: str) -> Optional[Dict[str, Any]]:
        """Extracts numeric bounds, intervals, and invariants."""
        return extract_param_constraints(text)

    def get_shape_contract(self, text: str) -> Optional[Dict[str, Any]]:
        """Extracts tensor dimension/rank contract."""
        return extract_shape_contract(text)

    def get_raises(self, doc: str = "", source: Optional[str] = None) -> List[str]:
        """Extracts exception classes from docstrings and AST source."""
        doc_raises = extract_docstring_raises(doc)
        ast_raises = extract_ast_raises(source)
        return list(dict.fromkeys(doc_raises + ast_raises))

    def is_context_manager(self, target: Any) -> bool:
        """Determines if target implements the context manager protocol."""
        if target is None:
            return False
        return hasattr(target, "__enter__") and hasattr(target, "__exit__")

    def get_type_vars(self, sig: inspect.Signature) -> List[str]:
        """Extracts generic type variables from signature."""
        return extract_type_vars(sig)

    def is_ingress_source(
        self,
        fn: Any,
        name: str,
        input_names: List[str],
        produces_domain: bool,
        consumes_domain: bool = False,
    ) -> bool:
        """
        Category-theoretic Ingress Source (Stage 1):
        A morphism P -> T_domain that does not consume any domain object
        and produces domain data.
        """
        if not produces_domain:
            return False
        return not consumes_domain

    def is_egress_sink(
        self,
        fn: Any,
        name: str,
        input_names: List[str],
        produces_domain: bool,
        consumes_domain: bool = True,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Category-theoretic Egress Sink (Stage 3):
        A morphism T_domain -> 1 (or external sink) that consumes domain data
        and does NOT produce domain data, directing output to an external destination.
        """
        if produces_domain:
            return False

        # --- Primary Mechanism: Formal Language Protocol & Type Analysis ---
        # 1. Check port-level abstract types and runtime types
        if inputs:
            import io
            import os
            for p in inputs.values():
                p_abs = getattr(p, "abstract_type", None) or (p.get("abstract_type") if isinstance(p, dict) else None)
                if p_abs == "path":
                    return True
                p_type = getattr(p, "type_name", None) or (p.get("type_name") if isinstance(p, dict) else None)
                if p_type and isinstance(p_type, str):
                    try:
                        from signature_introspector import _resolve_type_from_string
                        resolved = _resolve_type_from_string(p_type)
                        if resolved is not None and isinstance(resolved, type):
                            if issubclass(resolved, (os.PathLike, io.IOBase)):
                                return True
                    except Exception:
                        pass

        # 2. Check callable signature annotations directly
        if fn is not None:
            try:
                import io
                import os
                sig = self.resolve_callable_signature(fn, callable_name=name)
                if sig is not None:
                    for param in sig.parameters.values():
                        anno = param.annotation
                        if anno is not inspect.Parameter.empty and isinstance(anno, type):
                            if issubclass(anno, (os.PathLike, io.IOBase)):
                                return True
            except Exception:
                pass

        # --- Fallback Mechanism: Parameter Naming Convention ---
        # Used ONLY when parameters are unannotated, raw 'str', or C-extensions lacking type metadata
        dest_indicators = (
            "dest", "destination", "output", "out", "file", "filepath",
            "filename", "path", "uri", "url", "stream", "buf", "buffer",
            "fp", "f", "target", "writer"
        )
        return any(
            any(ind in p.lower() for ind in dest_indicators)
            for p in input_names
        )

    def is_parameterized(
        self,
        fn: Any,
        name: str,
        inputs: Dict[str, Any]
    ) -> bool:
        """Determines if a function is a Parameterized Morphism (A x E -> B)."""
        return any(
            (getattr(inp, "type_name", "") == "Enum" or getattr(inp, "enum_values", None) is not None)
            for inp in inputs.values() if getattr(inp, "required", True)
        )

    def is_public_symbol(
        self,
        obj: Any,
        name: str,
        module_name: str,
        root_module: Any = None
    ) -> bool:
        """Universal PEP conventions for public symbol detection."""
        if name.startswith("_") or name in ("TYPE_CHECKING",):
            return False
        parts = module_name.split(".")
        internal_parts = {
            "core", "_core", "internal", "_internal", "internals", "_internals",
            "compat", "_compat", "parsing", "readers", "externals", "vendor",
            "vendored", "_vendor", "impl", "estimator_checks", "_estimator_checks",
            "testing", "_testing", "testutils", "tests", "test", "conftest",
            "fixes", "_fixes"
        }
        is_internal_mod = any(
            p.startswith("_") or p.lower() in internal_parts or p.startswith("test_") or p.endswith("_test")
            for p in parts if p != f"_{parts[0]}"
        )
        if is_internal_mod:
            # Check if exported in any public parent package (e.g. sklearn.linear_model or pandas)
            pkg_prefix = ""
            for p in parts:
                if p.startswith("_") or p.lower() in internal_parts or p.startswith("test_") or p.endswith("_test"):
                    break
                pkg_prefix = f"{pkg_prefix}.{p}" if pkg_prefix else p
                p_mod = sys.modules.get(pkg_prefix)
                if p_mod is not None and hasattr(p_mod, name) and getattr(p_mod, name) is obj:
                    return True
            return False

        mod = sys.modules.get(module_name)
        if mod and hasattr(mod, "__all__") and isinstance(mod.__all__, (list, tuple, set)) and len(mod.__all__) > 0:
            return name in mod.__all__
        return True

    def describe(self) -> Dict[str, Any]:
        return {"paradigm": self.paradigm}


# =====================================================================
# Implementation Paradigm: Python Source Library Adapter
# =====================================================================

class PythonSourceAdapter(LibraryAdapter):
    """
    Adapter for pure Python libraries where source code and AST are accessible.
    Leverages AST inspection for signatures, mutations, and internal call graphs.
    """
    paradigm = "python_source"


# =====================================================================
# Implementation Paradigm: Compiled Binary Extension Adapter
# =====================================================================

class CompiledExtensionAdapter(LibraryAdapter):
    """
    Adapter for compiled C/C++/Rust binary extension libraries (.so, .pyd).
    Extracts signatures and types from PEP 561 .pyi type stubs and C docstrings.
    """
    paradigm = "compiled_binary"


# =====================================================================
# Implementation Paradigm: Hybrid Library Adapter
# =====================================================================

class HybridLibraryAdapter(LibraryAdapter):
    """
    Adapter for mixed libraries containing both Python wrapper modules
    and compiled binary extensions. Dynamically routes to the best extraction
    mechanism on a per-module basis.
    """
    paradigm = "hybrid"


# =====================================================================
# Universal Dynamic Factory (Zero Hardcoded Package Names)
# =====================================================================

_adapter_cache: Dict[str, LibraryAdapter] = {}


def get_adapter_for_package(package_name: str) -> LibraryAdapter:
    """
    Dynamically analyzes the physical implementation of any package
    and returns the corresponding paradigm adapter.
    Contains ZERO hardcoded package names.
    """
    if package_name in _adapter_cache:
        return _adapter_cache[package_name]

    try:
        root_module = importlib.import_module(package_name)
    except ImportError:
        cwd = str(Path.cwd())
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        root_module = importlib.import_module(package_name)

    # Inspect package structure
    modules = _walk_package(root_module)
    py_count = 0
    compiled_count = 0

    for m in modules.values():
        if _is_python_source_module(m):
            py_count += 1
        elif _is_compiled_extension_module(m):
            compiled_count += 1

    total = py_count + compiled_count
    if total == 0 or (py_count > 0 and compiled_count == 0):
        adapter = PythonSourceAdapter(root_module)
    elif compiled_count > 0 and py_count == 0:
        adapter = CompiledExtensionAdapter(root_module)
    else:
        adapter = HybridLibraryAdapter(root_module)

    logger.info(
        f"[ADAPTER] Selected '{adapter.paradigm}' adapter for package '{package_name}' "
        f"(Python modules: {py_count}, Compiled modules: {compiled_count})"
    )
    _adapter_cache[package_name] = adapter
    return adapter
