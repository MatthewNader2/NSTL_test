"""
src/gevr_sandbox.py - Neuro-Symbolic Topological Lattice (NSTL)
Generate-Execute-Verify-Repair (GEVR) Execution Sandbox.
Executes candidate scripts in persistent worker processes for sub-100ms latency.
"""

from __future__ import annotations
import ast
import contextlib
import io
import multiprocessing
import os
import sys
import threading
import traceback
import re
from typing import Tuple, Optional, Callable, Dict, Any, Union, List

try:
    from typing import NamedTuple
except ImportError:  # pragma: no cover
    from collections import namedtuple as NamedTuple  # type: ignore

from log_config import get_logger

try:
    from .config import settings
    from .utils import extract_code_from_llm_response
    from .lattice import TypeRegistry
except (ImportError, ValueError):
    from config import settings
    from utils import extract_code_from_llm_response
    from lattice import TypeRegistry

logger = get_logger('gevr_sandbox')


class ExecutionResult(NamedTuple):
    """
    Structured result of a sandbox execution.

    Backwards compatible with the historical 3-tuple contract
    ``(success, stdout, error)`` — tuple unpacking and indexing still work —
    while additionally exposing named attributes for programmatic callers:

        res = sandbox.execute_and_verify(code)
        res.success / res.verified      # bool
        res.stdout / res.error_message  # str
    """
    success: bool
    stdout: str
    error: str

    @property
    def verified(self) -> bool:
        """Alias for :attr:`success` (GEVR verification passed)."""
        return self.success

    @property
    def stderr(self) -> str:
        """Alias for :attr:`error` (the captured error channel)."""
        return self.error

    @property
    def error_message(self) -> str:
        """Alias for :attr:`error` (empty string on success)."""
        return self.error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "verified": self.success,
            "stdout": self.stdout,
            "stderr": self.error,
            "error": self.error,
            "error_message": self.error,
        }


class RepairResult(NamedTuple):
    """
    Structured result of a GEVR repair cycle.

    Backwards compatible with the historical 3-tuple ``(success, code, error)``;
    ``stdout`` is exposed as an alias of ``error`` (the diagnostic channel).
    """
    success: bool
    code: str
    error: str

    @property
    def verified(self) -> bool:
        return self.success

    @property
    def error_message(self) -> str:
        return self.error


def _init_worker(paths: list[str]):
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)


_BLOCKED_MODULES = frozenset({
    'subprocess', 'shutil', 'socket', 'ctypes',
    'signal', 'importlib', 'multiprocessing', 'threading', 'http',
    'urllib', 'ftplib', 'smtplib', 'telnetlib', 'xmlrpc', 'code',
    'codeop', 'compileall', 'py_compile', 'zipimport', 'pkgutil',
})

def _restricted_import(name, *args, **kwargs):
    base = name.split('.')[0]
    if base in _BLOCKED_MODULES:
        raise ImportError(f"Import of '{name}' is blocked in NSTL sandbox for security.")
    return __builtins__.__import__(name, *args, **kwargs) if hasattr(__builtins__, '__import__') else __import__(name, *args, **kwargs)

try:
    from .errors import DataflowExecutionError, ArtifactMaterializationError, PostconditionVerificationError
except (ImportError, ValueError):
    from errors import DataflowExecutionError, ArtifactMaterializationError, PostconditionVerificationError


def _check_estimator_fitted(estimator: Any, cell_id: str, var_name: str) -> None:
    """
    Verifies that an estimator object has been fitted — via LIBRARY-LEVEL
    PROTOCOLS, not engine-side attribute enumeration:
      1. The standard fitted-estimator protocol (`__sklearn_is_fitted__`,
         implemented by every conforming estimator) when available.
      2. The universal fitted-attribute convention of the estimator API:
         attributes set by `.fit()` end in a single trailing underscore
         (documented library-wide convention; any library following it
         verifies with zero engine-side per-class lists).
    """
    if estimator is None:
        raise PostconditionVerificationError(f"Estimator '{var_name}' ({cell_id}) is None.")

    # 1. Library protocol (duck-typed: no library import in engine code)
    protocol = getattr(estimator, "__sklearn_is_fitted__", None)
    if callable(protocol):
        try:
            if protocol():
                return
        except Exception as e:
            raise PostconditionVerificationError(
                f"Estimator '{var_name}' ({cell_id}) failed its fitted-protocol check: {e}"
            )

    # 2. Universal trailing-underscore fitted-attribute convention.
    # Instance attributes only (vars), excluding dunder and hyperparameters
    # (hyperparameters do NOT end in underscore; fitted attributes do).
    for attr_name, attr_val in vars(estimator).items():
        if attr_name.startswith("_"):
            continue
        if attr_name.endswith("_") and not attr_name.endswith("__"):
            if attr_val is not None:
                return

    raise PostconditionVerificationError(
        f"Estimator '{var_name}' ({cell_id}) has not been fitted (.fit() was not called or failed)."
    )


def _evaluate_cell_postcondition(exec_globals: Dict[str, Any], check: Dict[str, Any]) -> None:
    """Evaluates an individual Phase-1 postcondition against runtime execution scope."""
    cell_id = check.get("cell_id", "unknown_cell")
    target_var = check.get("target_var")
    prop = check.get("property")
    op = check.get("operator", "==")
    val = check.get("value")
    expr = check.get("expression")
    desc = check.get("description") or expr or f"{prop} {op} {val}"

    if not target_var or target_var not in exec_globals:
        return

    target_val = exec_globals[target_var]

    # Raw expression evaluation
    if expr:
        # State predicates are evaluated through the declared-property
        # evaluator below (the token `state` is not a runtime variable).
        is_state_predicate = bool(
            re.search(r"\bstate\s*(?:==|!=|is)\b", expr)
        )
        if not is_state_predicate:
            subbed_expr = expr
            target_name = check.get("target_port") or check.get("target") or "output_var"
            if target_name in subbed_expr:
                subbed_expr = re.sub(rf"\b{re.escape(target_name)}\b", target_var, subbed_expr)

            eval_scope = {
                "__builtins__": {
                    "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr,
                    "len": len, "type": type, "bool": bool, "int": int, "float": float,
                    "str": str, "True": True, "False": False, "None": None
                },
                target_var: target_val
            }
            import types
            for k, v in exec_globals.items():
                if isinstance(v, types.ModuleType) or (not k.startswith("__") and k != target_var):
                    eval_scope[k] = v

            try:
                passed = bool(eval(subbed_expr, eval_scope, exec_globals))
            except PostconditionVerificationError:
                raise
            except Exception as e:
                # FAIL CLOSED: a postcondition that cannot be evaluated is a
                # verification failure, never a silent pass.
                raise PostconditionVerificationError(
                    f"Postcondition for cell '{cell_id}' could not be evaluated: '{expr}' raised {type(e).__name__}: {e}"
                )
            if not passed:
                raise PostconditionVerificationError(
                    f"Postcondition failed for cell '{cell_id}': '{desc}' evaluated to False on variable '{target_var}'."
                )

    # Property-based evaluation
    if prop == "ndim":
        actual_ndim = getattr(target_val, "ndim", None)
        if actual_ndim is not None:
            if op == "==" and actual_ndim != val:
                raise PostconditionVerificationError(
                    f"Postcondition 'ndim == {val}' failed for cell '{cell_id}': '{target_var}.ndim' is {actual_ndim}."
                )
            elif op == ">=" and actual_ndim < val:
                raise PostconditionVerificationError(
                    f"Postcondition 'ndim >= {val}' failed for cell '{cell_id}': '{target_var}.ndim' is {actual_ndim}."
                )
    elif prop == "has_nans":
        has_nans = False
        if hasattr(target_val, "isna"):
            has_nans = bool(target_val.isna().any().any() if hasattr(target_val.isna().any(), "any") else target_val.isna().any())
        elif hasattr(target_val, "isnull"):
            has_nans = bool(target_val.isnull().any().any() if hasattr(target_val.isnull().any(), "any") else target_val.isnull().any())
        elif hasattr(target_val, "dtype") and getattr(target_val, "size", 0) > 0:
            import numpy as _np
            has_nans = bool(_np.isnan(target_val).any()) if _np.issubdtype(target_val.dtype, _np.number) else False

        expected_has_nans = bool(val)
        if has_nans != expected_has_nans:
            raise PostconditionVerificationError(
                f"Postcondition 'has_nans == {expected_has_nans}' failed for cell '{cell_id}': variable '{target_var}' {'contains NaNs' if has_nans else 'does not contain NaNs'}."
            )
    elif prop == "is_deduped":
        if hasattr(target_val, "duplicated"):
            is_deduped = not bool(target_val.duplicated().any())
            if is_deduped != bool(val):
                raise PostconditionVerificationError(
                    f"Postcondition 'is_deduped == {val}' failed for cell '{cell_id}': DataFrame '{target_var}' contains duplicates."
                )
    elif prop == "is_fitted":
        _check_estimator_fitted(target_val, cell_id, target_var)
    elif prop == "state":
        # Declared-property evaluation: the expected state name is looked up in
        # the TypeRegistry's DECLARED typestate vocabulary (trees declare
        # carrier_type + verifiable properties per state). Shape logic comes
        # from the declaration, not from engine-side domain knowledge.
        registry = TypeRegistry.get_instance()
        state_props = registry.get_state_properties(str(val))
        carrier = registry.get_state_carrier(str(val))

        violations = []
        shape = getattr(target_val, "shape", None)
        ndim = getattr(target_val, "ndim", None)

        channels = state_props.get("channels")
        if channels is not None and shape is not None and len(shape) >= 2:
            actual_channels = shape[2] if len(shape) == 3 else 1
            if actual_channels != channels:
                violations.append(
                    f"declared {channels} channel(s), variable has shape {shape}"
                )

        expect_ndim = state_props.get("ndim")
        if expect_ndim is not None and ndim is not None and int(expect_ndim) != int(ndim):
            violations.append(f"declared ndim={expect_ndim}, variable has ndim={ndim}")

        dtype_decl = state_props.get("dtype")
        if dtype_decl and hasattr(target_val, "dtype"):
            actual_dtype = str(getattr(target_val.dtype, "name", target_val.dtype))
            if str(dtype_decl).lower() not in (actual_dtype.lower(), ""):
                violations.append(f"declared dtype={dtype_decl}, variable has dtype={actual_dtype}")

        if violations:
            raise PostconditionVerificationError(
                f"Postcondition 'state == {val}' failed for cell '{cell_id}': variable '{target_var}' "
                + "; ".join(violations) + "."
            )


def _evaluate_terminal_intent(
    exec_globals: Dict[str, Any],
    term_check: Dict[str, Any],
    egress_paths: Optional[list[str]] = None
) -> None:
    """Evaluates task intent on terminal nodes (model fit split data, image annotation egress, etc.)."""
    intent_type = term_check.get("type")
    cell_id = term_check.get("cell_id", "terminal_node")

    if intent_type == "model_fit_split":
        model_var = term_check.get("model_var")
        feature_var = term_check.get("feature_var")
        expected_train_var = term_check.get("expected_train_feature_var")
        unsplit_feature_var = term_check.get("unsplit_feature_var")

        if not model_var or model_var not in exec_globals:
            raise PostconditionVerificationError(
                f"Terminal model variable '{model_var}' was not created in execution scope."
            )
        model_obj = exec_globals[model_var]
        _check_estimator_fitted(model_obj, cell_id, model_var)

        # Verify model was fitted on split training partition (not full unsplit dataset)
        if expected_train_var and feature_var:
            if feature_var != expected_train_var:
                raise PostconditionVerificationError(
                    f"Terminal model '{model_var}' ({cell_id}) was fitted on '{feature_var}' instead of split training partition '{expected_train_var}'. Model training intent violated."
                )

        if unsplit_feature_var and expected_train_var:
            unsplit_val = exec_globals.get(unsplit_feature_var)
            train_val = exec_globals.get(expected_train_var)
            if unsplit_val is not None and train_val is not None:
                n_unsplit = len(unsplit_val) if hasattr(unsplit_val, "__len__") else 0
                n_train = len(train_val) if hasattr(train_val, "__len__") else 0
                if n_train > 0 and n_train < n_unsplit:
                    # Library-agnostic probe: a fitted model that exposes its
                    # consumed training-sample count (via any declared fitted
                    # attribute following the trailing-underscore convention)
                    # must NOT report the un-split row count.
                    n_seen = None
                    for attr_name, attr_val in vars(model_obj).items():
                        if attr_name.startswith("_") or not attr_name.endswith("_") or attr_name.endswith("__"):
                            continue
                        if isinstance(attr_val, (int, float)):
                            candidate = attr_val
                        elif hasattr(attr_val, "shape") and getattr(attr_val.shape, "__len__", lambda: 0)() > 0:
                            candidate = attr_val.shape[0]
                        elif hasattr(attr_val, "__len__"):
                            try:
                                candidate = len(attr_val[0]) if len(attr_val) and hasattr(attr_val[0], "__len__") else len(attr_val)
                            except Exception:
                                continue
                        else:
                            continue
                        n_seen = candidate
                        break
                    if n_seen is not None and n_seen == n_unsplit:
                        raise PostconditionVerificationError(
                            f"Terminal model '{model_var}' ({cell_id}) was fitted on un-split dataset ({n_unsplit} samples) instead of training partition ({n_train} samples)."
                        )

    elif intent_type == "image_annotation_egress":
        saved_var = term_check.get("saved_var")
        annotated_var = term_check.get("annotated_var")
        ingress_var = term_check.get("ingress_var")
        output_path = term_check.get("output_path", "")

        # 1. Wire check: verify saved variable is annotated wire, not raw ingress wire
        if annotated_var and ingress_var and saved_var:
            if saved_var == ingress_var and annotated_var != ingress_var:
                raise PostconditionVerificationError(
                    f"Terminal node '{cell_id}' saved unannotated raw input image '{ingress_var}' instead of annotated image '{annotated_var}'. Annotation intent was not realized in egress artifact."
                )

        # 2. Disk content check: verify saved artifact is not identical to raw input
        clean_path = output_path.strip("'\"") if output_path else None
        if not clean_path and egress_paths:
            clean_path = egress_paths[0].strip("'\"") if egress_paths else None

        if clean_path and os.path.exists(clean_path):
            # Artifact readers: the sandbox must decode the on-disk artifact to
            # compare it against the ingress wire. Library choice here is a
            # pluggable mechanism (declare a reader in the domain tree to
            # override); it carries no routing/domain vocabulary.
            import cv2
            import numpy as _np
            disk_img = cv2.imread(clean_path)
            ingress_img = exec_globals.get(ingress_var) if ingress_var else None

            if disk_img is not None and ingress_img is not None:
                if disk_img.shape == ingress_img.shape and _np.array_equal(disk_img, ingress_img):
                    if annotated_var and annotated_var != ingress_var:
                        raise PostconditionVerificationError(
                            f"Egress image '{clean_path}' contains byte-identical raw input image without annotations. Annotation intent was not realized in saved artifact."
                        )

    elif intent_type == "tabular_egress":
        saved_var = term_check.get("saved_var")
        ingress_var = term_check.get("ingress_var")
        output_path = term_check.get("output_path", "")
        expected_clean = term_check.get("expected_clean", {})

        if saved_var and ingress_var and saved_var == ingress_var and expected_clean.get("no_nans"):
            raise PostconditionVerificationError(
                f"Terminal node '{cell_id}' saved raw uncleaned DataFrame '{ingress_var}' instead of transformed DataFrame."
            )

        clean_path = output_path.strip("'\"") if output_path else None
        if not clean_path and egress_paths:
            clean_path = egress_paths[0].strip("'\"") if egress_paths else None

        if clean_path and os.path.exists(clean_path) and expected_clean.get("no_nans"):
            # Tabular artifact reader (see note above on pluggable mechanisms).
            import pandas as _pd
            try:
                disk_df = _pd.read_csv(clean_path)
                if disk_df.isna().any().any():
                    raise PostconditionVerificationError(
                        f"Egress CSV '{clean_path}' contains NaN values; dropna intent was not realized in saved artifact."
                    )
            except Exception as e:
                if isinstance(e, PostconditionVerificationError):
                    raise

    elif intent_type == "visualization_egress":
        fig_var = term_check.get("fig_var")
        output_path = term_check.get("output_path", "")

        fig_obj = exec_globals.get(fig_var) if fig_var else None
        if fig_obj is None:
            for v in exec_globals.values():
                if type(v).__name__ == "Figure":
                    fig_obj = v
                    break

        if fig_obj is not None:
            axes = getattr(fig_obj, "axes", [])
            total_elements = sum(
                len(getattr(ax, "lines", [])) +
                len(getattr(ax, "collections", [])) +
                len(getattr(ax, "patches", [])) +
                len(getattr(ax, "containers", [])) +
                len(getattr(ax, "images", []))
                for ax in axes
            )
            if len(axes) > 0 and total_elements == 0:
                raise PostconditionVerificationError(
                    f"Terminal node '{cell_id}' saved an empty figure to '{output_path}' with 0 plotted data elements."
                )


def verify_postconditions(
    exec_globals: Dict[str, Any],
    verification_spec: Any,
    egress_paths: Optional[list[str]] = None
) -> None:
    """
    Evaluates Phase-1 postconditions and terminal node intent against runtime execution scope.
    Raises PostconditionVerificationError if any check fails.
    """
    if verification_spec is None:
        return

    if hasattr(verification_spec, "verify") and callable(verification_spec.verify):
        verification_spec.verify(exec_globals, egress_paths)
        return

    if isinstance(verification_spec, dict):
        cell_checks = verification_spec.get("cell_checks", [])
        terminal_checks = verification_spec.get("terminal_checks", [])
    elif isinstance(verification_spec, list):
        cell_checks = verification_spec
        terminal_checks = []
    else:
        return

    # 1. Evaluate Phase-1 Cell Postconditions
    for check in cell_checks:
        _evaluate_cell_postcondition(exec_globals, check)

    # 2. Evaluate Terminal Node Intent Checks
    for term_check in terminal_checks:
        _evaluate_terminal_intent(exec_globals, term_check, egress_paths)


def _sandbox_worker_exec(
    code: str,
    egress_paths: Optional[list[str]] = None,
    cwd: Optional[str] = None,
    verification_spec: Optional[Union[Dict[str, Any], List[Dict[str, Any]], Any]] = None
) -> Dict[str, Any]:
    """
    Isolated execution unit executed within persistent worker process.
    Executes the candidate strictly as-is: no synthetic fixtures are ever created.
    Missing input artifacts surface as honest FileNotFoundError failures, and only
    artifacts the code itself materializes at declared egress destinations count
    towards the physical verification pass.
    """
    if cwd:
        try:
            os.chdir(cwd)
        except Exception:
            pass
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        try:
            builtins_dict = dict(__builtins__.__dict__) if hasattr(__builtins__, "__dict__") else dict(__builtins__)
            builtins_dict["__import__"] = _restricted_import
            exec_globals: Dict[str, Any] = {
                "__name__": "__main__",
                "__builtins__": builtins_dict
            }
            exec(code, exec_globals)

            # 1. Dataflow Non-Vacuity Verification: inspect top-level AST for final assigned variable
            parsed = ast.parse(code)
            terminal_var = None
            for node in reversed(parsed.body):
                if isinstance(node, ast.Assign):
                    for target in reversed(node.targets):
                        if isinstance(target, ast.Name):
                            terminal_var = target.id
                            break
                    if terminal_var:
                        break
                elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                    if isinstance(node.target, ast.Name):
                        terminal_var = node.target.id
                        break

            if terminal_var:
                if terminal_var not in exec_globals:
                    raise DataflowExecutionError(
                        f"Terminal pipeline variable '{terminal_var}' was not created in execution scope."
                    )
                val = exec_globals[terminal_var]
                if val is None:
                    raise DataflowExecutionError(
                        f"Terminal pipeline variable '{terminal_var}' evaluated to None."
                    )
                if val is False and egress_paths:
                    raise DataflowExecutionError(
                        f"Terminal pipeline save operation '{terminal_var}' returned False."
                    )

            # 2. Guarded Branch Failure Detection in STDERR
            stdout_str = stdout_buf.getvalue()
            stderr_str = stderr_buf.getvalue()
            stderr_lower = stderr_str.lower()
            error_markers = [
                "traceback (most recent call last)", "segmentation fault",
                "fatal error", "core dumped"
            ]
            for marker in error_markers:
                if marker in stderr_lower:
                    raise DataflowExecutionError(
                        f"Guarded failure detected in stderr: '{marker}'."
                    )

            # 3. Physical Artifact Materialization Verification
            if egress_paths:
                for p in egress_paths:
                    if not p:
                        continue
                    clean_p = p.strip("\"'")
                    if not os.path.exists(clean_p) or os.path.getsize(clean_p) == 0:
                        raise ArtifactMaterializationError(
                            f"Egress destination artifact '{clean_p}' was not created or has 0 bytes."
                        )

            # 4. Phase-1 Postcondition & Terminal Task Verification
            if verification_spec:
                verify_postconditions(exec_globals, verification_spec, egress_paths)

            return {
                "success": True,
                "stdout": stdout_str,
                "stderr": stderr_str,
                "error": ""
            }
        except Exception as e:
            is_extrinsic = isinstance(e, (FileNotFoundError, ConnectionError, TimeoutError, ModuleNotFoundError))
            err_msg = f"{type(e).__name__}: {e}" if isinstance(e, (DataflowExecutionError, ArtifactMaterializationError, PostconditionVerificationError)) else traceback.format_exc()
            return {
                "success": False,
                "extrinsic": is_extrinsic,
                "stdout": stdout_buf.getvalue(),
                "stderr": stderr_buf.getvalue(),
                "error": err_msg
            }
        finally:
            pass


class GEVRSandbox:
    """
    Sandboxed Python execution engine with persistent worker pool for sub-50ms execution.
    """
    _pool: Optional[multiprocessing.Pool] = None
    _lock = threading.Lock()

    def __init__(self, num_workers: int = None, timeout_seconds: float = None):
        self.timeout = timeout_seconds if timeout_seconds is not None else settings.sandbox_timeout
        self._ensure_pool(num_workers if num_workers is not None else settings.sandbox_workers)

    @classmethod
    def _ensure_pool(cls, num_workers: int = 2):
        if cls._pool is None:
            with cls._lock:
                if cls._pool is None:
                    ctx_name = "spawn"
                    ctx = multiprocessing.get_context(ctx_name)
                    cls._pool = ctx.Pool(
                        processes=num_workers,
                        initializer=_init_worker,
                        initargs=(list(sys.path),),
                        maxtasksperchild=100
                    )

    def execute(
        self,
        code: str,
        timeout: Optional[float] = None,
        egress_paths: Optional[list[str]] = None,
        cwd: Optional[str] = None,
        verification_spec: Optional[Union[Dict[str, Any], List[Dict[str, Any]], Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes Python code in the persistent worker process pool.
        Returns: {'success': bool, 'stdout': str, 'stderr': str, 'error': str}
        """
        if not code or not code.strip():
            return {"success": False, "stdout": "", "stderr": "", "error": "ExecutionError: Empty code block."}

        # Pre-execution AST syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            return {"success": False, "stdout": "", "stderr": "", "error": f"SyntaxError: {e}"}

        tout = timeout if timeout is not None else self.timeout
        exec_cwd = cwd or os.getcwd()
        spec_dict = verification_spec.to_dict() if hasattr(verification_spec, "to_dict") else verification_spec

        try:
            self._ensure_pool()
            async_res = self._pool.apply_async(_sandbox_worker_exec, (code, egress_paths, exec_cwd, spec_dict))
            res = async_res.get(timeout=tout)
            if res.get("error") is None:
                res["error"] = ""
            return res
        except multiprocessing.TimeoutError:
            return {"success": False, "stdout": "", "stderr": "", "error": f"ExecutionTimedOut: Exceeded {tout}s execution limit."}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": "", "error": f"ExecutionSystemError: {e}"}

    def execute_and_verify(
        self,
        code: str,
        egress_paths: Optional[list[str]] = None,
        cwd: Optional[str] = None,
        verification_spec: Optional[Union[Dict[str, Any], List[Dict[str, Any]], Any]] = None
    ) -> ExecutionResult:
        """
        Executes Python code and verifies it in the sandbox.

        Returns an :class:`ExecutionResult` — a named 3-tuple
        ``(success, stdout, error)``. Callers may either unpack it positionally
        (``success, stdout, err = sandbox.execute_and_verify(code)``) or access
        the named fields (``res.verified``, ``res.error_message``).
        """
        res = self.execute(code, egress_paths=egress_paths, cwd=cwd, verification_spec=verification_spec)
        err = res.get("error", "") or res.get("stderr", "")
        return ExecutionResult(bool(res["success"]), res.get("stdout", ""), err)

    def repair_cycle(
        self,
        initial_code: str,
        llm_repair_func: Optional[Callable[[str, str], str]] = None,
        max_attempts: int = 2,
        egress_paths: Optional[list[str]] = None,
        cwd: Optional[str] = None,
        verification_spec: Optional[Union[Dict[str, Any], List[Dict[str, Any]], Any]] = None
    ) -> RepairResult:
        """
        Feedback verification loop:
        Executes code, captures tracebacks and task verification errors, and applies diagnostic LLM repairs.

        Returns a :class:`RepairResult` — a named 3-tuple ``(success, code, error)``
        where ``code`` is the final (possibly repaired) code and ``error`` the last
        diagnostic message (empty on success).
        """
        current_code = initial_code
        error = ""
        for attempt in range(max_attempts):
            exec_res = self.execute_and_verify(
                current_code, egress_paths=egress_paths, cwd=cwd, verification_spec=verification_spec
            )
            success, stdout, error = exec_res.success, exec_res.stdout, exec_res.error
            if success:
                logger.info(f"[GEVR Sandbox] Verification PASSED on attempt {attempt + 1}")
                return RepairResult(True, current_code, "")

            logger.warning(f"[GEVR Sandbox] Attempt {attempt + 1} failed with error:\n{error}")
            if attempt < max_attempts - 1:
                if llm_repair_func:
                    logger.info(f"[GEVR Sandbox] Requesting repair heuristic for attempt {attempt + 2}...")
                    repaired = extract_code_from_llm_response(llm_repair_func(current_code, error))
                    if repaired and repaired != current_code:
                        current_code = repaired
                    else:
                        break
                else:
                    try:
                        from .unification import UnificationGate
                    except (ImportError, ValueError):
                        from unification import UnificationGate
                    repaired = UnificationGate.resolve_imports(current_code)
                    if repaired and repaired != current_code:
                        current_code = repaired
                    else:
                        break
            else:
                break

        return RepairResult(False, current_code, error)
