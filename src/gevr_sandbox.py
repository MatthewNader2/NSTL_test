"""
src/gevr_sandbox.py
Hardened, isolated, domain-agnostic sandboxed execution environment with
AST security inspection, zero worker leakage, and pluggable verification protocols.
"""

from __future__ import annotations

import abc
import ast
import builtins
import contextlib
import logging
import multiprocessing
import os
import queue as _pyqueue
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Security Exceptions & AST Validator
# ============================================================================

class SecurityViolationError(PermissionError):
    """Raised when executed code violates sandbox security constraints."""
    pass


class SandboxTimeoutError(TimeoutError):
    """Raised when execution exceeds the allocated wall-clock time limit."""
    pass


class _SecurityASTVisitor(ast.NodeVisitor):
    """
    Static analysis pass over synthesized code before execution.
    Detects dunder traversal exploits, bytecode manipulation, and forbidden builtins.
    """

    FORBIDDEN_ATTRIBUTES: frozenset[str] = frozenset({
        "__subclasses__",
        "__bases__",
        "__mro__",
        "__globals__",
        "__code__",
        "__closure__",
        "__builtins__",
        "__import__",
        "gi_frame",
        "f_locals",
        "f_globals",
        "cr_frame",
    })

    FORBIDDEN_CALLS: frozenset[str] = frozenset({
        "eval",
        "exec",
        "compile",
        "breakpoint",
    })

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in self.FORBIDDEN_ATTRIBUTES:
            raise SecurityViolationError(
                f"Access to restricted meta-attribute '{node.attr}' at line {node.lineno} is forbidden."
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id in self.FORBIDDEN_CALLS:
            raise SecurityViolationError(
                f"Direct access to forbidden built-in '{node.id}' at line {node.lineno} is blocked."
            )
        self.generic_visit(node)


# ============================================================================
# Pluggable Postconditions & State Protocols (Domain-Agnostic)
# ============================================================================

class StateProtocolRegistry:
    """Registry determining object readiness or fittedness via pluggable predicates."""

    def __init__(self):
        self._predicates: List[Callable[[Any], bool]] = []
        self._register_default_predicates()

    def register(self, predicate: Callable[[Any], bool]) -> None:
        self._predicates.append(predicate)

    def is_fitted_or_ready(self, target: Any) -> bool:
        """Check if target satisfies any registered state predicate."""
        for pred in self._predicates:
            try:
                if pred(target):
                    return True
            except Exception:
                continue
        return False

    def _register_default_predicates(self) -> None:
        # Standard protocol: Check for is_fitted(), is_ready(), or is_converged() methods
        def _callable_check(obj: Any) -> bool:
            for attr in ("is_fitted", "is_ready", "is_converged"):
                method = getattr(obj, attr, None)
                if callable(method):
                    return bool(method())
            return False

        # Generic state inspection: check for state-denoting attributes
        def _attribute_convention_check(obj: Any) -> bool:
            if hasattr(obj, "__dict__"):
                # Detect populated state without assuming third-party libraries
                public_state = [k for k in obj.__dict__.keys() if not k.startswith("_")]
                return len(public_state) > 0
            return True

        self.register(_callable_check)
        self.register(_attribute_convention_check)


class PostconditionRegistry:
    """Domain-agnostic registry for evaluating postcondition constraints."""

    def __init__(self):
        self._evaluators: Dict[str, Callable[[Any, Any], bool]] = {}
        self._register_default_evaluators()

    def register(self, property_name: str, evaluator: Callable[[Any, Any], bool]) -> None:
        self._evaluators[property_name] = evaluator

    def evaluate(self, property_name: str, target_val: Any, expected_val: Any) -> bool:
        evaluator = self._evaluators.get(property_name)
        if not evaluator:
            logger.warning(f"No postcondition evaluator registered for '{property_name}'. Falling back to equality.")
            return bool(target_val == expected_val)
        try:
            return evaluator(target_val, expected_val)
        except Exception as e:
            logger.error(f"Error evaluating postcondition '{property_name}': {e}")
            return False

    def _register_default_evaluators(self) -> None:
        # Dimensionality check via duck-typing (works for lists, tuples, tensors, arrays)
        def _ndim_evaluator(val: Any, expected: Any) -> bool:
            if hasattr(val, "ndim"):
                return bool(val.ndim == expected)
            if hasattr(val, "shape"):
                return bool(len(val.shape) == expected)
            # Recursive depth inspection for nested iterables
            def _get_depth(obj: Any) -> int:
                if isinstance(obj, (list, tuple)) and obj:
                    return 1 + _get_depth(obj[0])
                return 0
            return _get_depth(val) == expected

        # Null / NaN check via duck-typing
        def _has_nans_evaluator(val: Any, expected: Any) -> bool:
            # Check for duck-typed .isna() or .isnull()
            for method_name in ("isna", "isnull"):
                method = getattr(val, method_name, None)
                if callable(method):
                    res = method()
                    has_nan = bool(getattr(res, "any", lambda: res)())
                    return has_nan == expected

            # Check for native float('nan') or nested elements
            try:
                import math
                if isinstance(val, float) and math.isnan(val):
                    return True == expected
                if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
                    has_nan = any(isinstance(x, float) and math.isnan(x) for x in val)
                    return has_nan == expected
            except Exception:
                pass
            return False == expected

        # Uniqueness / Deduplicated check via duck-typing
        def _dedup_evaluator(val: Any, expected: Any) -> bool:
            if hasattr(val, "duplicated"):
                dups = val.duplicated()
                is_unique = not bool(getattr(dups, "any", lambda: dups)())
                return is_unique == expected
            try:
                seq = list(val)
                return (len(seq) == len(set(seq))) == expected
            except TypeError:
                # Elements not hashable
                return True == expected

        self.register("ndim", _ndim_evaluator)
        self.register("has_nans", _has_nans_evaluator)
        self.register("is_deduped", _dedup_evaluator)


# ============================================================================
# Pluggable Terminal Intent Verifiers (Domain-Agnostic)
# ============================================================================

@dataclass
class IntentVerificationResult:
    valid: bool
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class BaseIntentVerifier(abc.ABC):
    @abc.abstractmethod
    def verify(self, output: Any, context: Dict[str, Any]) -> IntentVerificationResult:
        pass


class TerminalIntentRegistry:
    """Decoupled registry of verifiers validating terminal computation intent."""

    def __init__(self):
        self._verifiers: Dict[str, BaseIntentVerifier] = {}
        self._register_default_verifiers()

    def register(self, intent_type: str, verifier: BaseIntentVerifier) -> None:
        self._verifiers[intent_type] = verifier

    def verify(self, intent_type: str, output: Any, context: Dict[str, Any]) -> IntentVerificationResult:
        verifier = self._verifiers.get(intent_type)
        if not verifier:
            # Fallback verification: simple presence and non-None check
            is_valid = output is not None
            return IntentVerificationResult(
                valid=is_valid,
                reason="Default fallback: output is non-None" if is_valid else "Output is None",
            )
        return verifier.verify(output, context)

    def _register_default_verifiers(self) -> None:
        # Verifier for partition/split intent (e.g. dataset or collection splits)
        class PartitionIntentVerifier(BaseIntentVerifier):
            def verify(self, output: Any, context: Dict[str, Any]) -> IntentVerificationResult:
                if not isinstance(output, (tuple, list)):
                    return IntentVerificationResult(valid=False, reason="Split intent must produce a collection")
                expected_min = context.get("min_partitions", 2)
                if len(output) < expected_min:
                    return IntentVerificationResult(
                        valid=False,
                        reason=f"Expected at least {expected_min} partitions, received {len(output)}",
                    )
                return IntentVerificationResult(valid=True, reason="Valid partition structure")

        # Verifier for structured/tabular data transformation intent
        class StructuredDataVerifier(BaseIntentVerifier):
            def verify(self, output: Any, context: Dict[str, Any]) -> IntentVerificationResult:
                if output is None:
                    return IntentVerificationResult(valid=False, reason="Target structured output is None")
                # Duck-type column or length existence
                has_len = hasattr(output, "__len__")
                has_shape = hasattr(output, "shape")
                if not (has_len or has_shape):
                    return IntentVerificationResult(valid=False, reason="Output lacks dimension or sequence protocol")
                return IntentVerificationResult(valid=True, reason="Structured tabular contract verified")

        # Verifier for artifact rendering intent
        class VisualArtifactVerifier(BaseIntentVerifier):
            def verify(self, output: Any, context: Dict[str, Any]) -> IntentVerificationResult:
                if output is None:
                    return IntentVerificationResult(valid=False, reason="Artifact output is None")
                # Look for common rendering protocols without importing matplotlib
                has_render_hook = any(
                    hasattr(output, attr) for attr in ("canvas", "savefig", "render", "show", "to_image")
                )
                if has_render_hook or type(output).__name__ in ("Figure", "Image", "Plot"):
                    return IntentVerificationResult(valid=True, reason="Visual artifact verified")
                return IntentVerificationResult(valid=False, reason="Object lacks visual artifact rendering protocol")

        self.register("model_fit_split", PartitionIntentVerifier())
        self.register("tabular_egress", StructuredDataVerifier())
        self.register("image_annotation_egress", StructuredDataVerifier())
        self.register("visualization_egress", VisualArtifactVerifier())


# ============================================================================
# Hardened Worker Execution Layer
# ============================================================================

_BLOCKED_MODULES = frozenset({
    "subprocess",
    "shutil",
    "socket",
    "ctypes",
    "pty",
    "commands",
    "multiprocessing",
    "threading",
    "_thread",
})


def _create_sanitized_os():
    """Wrap standard `os` to disable shell execution and destructive file operations."""
    import os as _real_os

    class _SanitizedOS:
        def __getattr__(self, name: str) -> Any:
            blocked_ops = {
                "system", "popen", "spawn", "spawnl", "spawnle", "spawnlp", "spawnlpe",
                "spawnv", "spawnve", "spawnvp", "spawnvpe", "execv", "execve", "execvp",
                "execvpe", "execl", "execle", "execlp", "execlpe", "remove", "unlink",
                "rmdir", "removedirs", "chmod", "chown", "kill", "killpg"
            }
            if name in blocked_ops:
                raise SecurityViolationError(f"Operating system operation 'os.{name}' is prohibited in sandbox.")
            return getattr(_real_os, name)

    return _SanitizedOS()


def _restricted_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
    """Guarded import callback preventing module escape."""
    root_module = name.split(".")[0]
    if root_module in _BLOCKED_MODULES:
        raise SecurityViolationError(f"Import of restricted module '{root_module}' is blocked.")

    if root_module == "os":
        return _create_sanitized_os()

    # Safely retrieve underlying native import
    real_import = builtins.__import__
    return real_import(name, globals, locals, fromlist, level)


def _sandbox_worker_exec(
    code_str: str,
    context: Dict[str, Any],
    runtime_aliases: Dict[str, str],
    result_queue: multiprocessing.Queue,
) -> None:
    """
    Subprocess worker executing sandboxed code in strict process isolation.
    Exceptions, stdout and stderr are captured and returned safely via the IPC queue.
    """
    import io as _io

    captured_stdout = _io.StringIO()
    captured_stderr = _io.StringIO()
    try:
        # Step 1: Pre-execution AST Analysis
        parsed_ast = ast.parse(code_str)
        _SecurityASTVisitor().visit(parsed_ast)

        # Step 2: Construct Sanitized Execution Environment
        safe_builtins: Dict[str, Any] = {}
        for k, v in builtins.__dict__.items():
            if k not in _SecurityASTVisitor.FORBIDDEN_CALLS:
                safe_builtins[k] = v

        # Correctly bind restricted import into safe builtins dictionary
        safe_builtins["__import__"] = _restricted_import

        exec_globals: Dict[str, Any] = {
            "__builtins__": safe_builtins,
            "__name__": "__sandbox__",
            "__doc__": None,
        }
        exec_globals.update(context)

        # Step 3: Bytecode Execution under captured streams
        compiled = compile(parsed_ast, filename="<sandbox>", mode="exec")
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            exec(compiled, exec_globals)

        # Extract mutated context or return value; apply the pipeline's
        # declared runtime aliases (e.g. a prompt-declared egress identifier
        # "Z" aliases the terminal output variable) at the runtime namespace
        # level — source code itself is never rewritten to fake a sink.
        raw_results = {
            k: v for k, v in exec_globals.items()
            if not k.startswith("__") and k not in context
        }
        for alias_name, source_var in (runtime_aliases or {}).items():
            try:
                if source_var in raw_results:
                    raw_results[alias_name] = raw_results[source_var]
            except Exception:
                continue

        # Sanitize for IPC: modules, functions and other unpicklable objects
        # would be silently DROPPED by the queue feeder thread (a lost payload
        # looks identical to a crashed worker on the parent side). Replace
        # anything unpicklable with a descriptive summary.
        import pickle as _pickle

        def _ipc_safe(value: Any) -> Any:
            try:
                _pickle.dumps(value)
                return value
            except Exception:
                return f"<unpicklable:{type(value).__name__}>"

        extracted_results = {
            k: _ipc_safe(v) for k, v in raw_results.items()
        }

        result_queue.put({
            "success": True,
            "returncode": 0,
            "stdout": captured_stdout.getvalue(),
            "stderr": captured_stderr.getvalue(),
            "error": None,
            "results": extracted_results,
        })

        # Flush the queue feeder thread before interpreter exit so the
        # payload can never be lost to the daemon-thread teardown race.
        result_queue.close()
        result_queue.join_thread()

    except SecurityViolationError as sec_err:
        result_queue.put({
            "success": False,
            "returncode": 1,
            "stdout": captured_stdout.getvalue(),
            "stderr": captured_stderr.getvalue(),
            "error": f"SecurityViolation: {str(sec_err)}",
            "results": {},
        })

        # Flush the queue feeder thread before interpreter exit so the
        # payload can never be lost to the daemon-thread teardown race.
        result_queue.close()
        result_queue.join_thread()
    except Exception as exc:
        formatted_exc = traceback.format_exc()
        result_queue.put({
            "success": False,
            "returncode": 1,
            "stdout": captured_stdout.getvalue(),
            "stderr": captured_stderr.getvalue(),
            "error": f"{type(exc).__name__}: {str(exc)}\n{formatted_exc}",
            "results": {},
        })

        # Flush the queue feeder thread before interpreter exit so the
        # payload can never be lost to the daemon-thread teardown race.
        result_queue.close()
        result_queue.join_thread()


# ============================================================================
# Main GEVR Sandbox Manager
# ============================================================================

class GEVRSandbox:
    """
    Hardened execution sandbox with strict process isolation, timeout enforcement,
    and automatic cleanup of hung processes.
    """

    def __init__(
        self,
        default_timeout: float = 5.0,
        postcondition_registry: Optional[PostconditionRegistry] = None,
        intent_registry: Optional[TerminalIntentRegistry] = None,
        state_registry: Optional[StateProtocolRegistry] = None,
    ):
        self.default_timeout = default_timeout
        self.postconditions = postcondition_registry or PostconditionRegistry()
        self.intents = intent_registry or TerminalIntentRegistry()
        self.state_protocols = state_registry or StateProtocolRegistry()

    # ------------------------------------------------------------------
    # Unified execution contract (Solution 4)
    # ------------------------------------------------------------------
    # One stable, caller-agnostic result envelope shared by CLI, API and Dev
    # Mode:
    #   {
    #     "success":      bool,   code ran AND satisfied egress/spec checks
    #     "returncode":   int,    worker exit code (0 == clean run)
    #     "stdout":       str,    captured standard output
    #     "stderr":       str,    captured standard error
    #     "error":        str,    "" when none
    #     "extrinsic":    bool,   environmental failure (not the code's fault)
    #     "results":      dict,   extracted global namespace (aliases applied)
    #     "egress":       dict,   path -> verification detail
    #   }
    # Callers previously crashed with TypeError (unexpected kwarg
    # 'egress_paths') or silently mis-read the raw globals dict for a
    # "success" key; this contract makes every entrypoint agree.
    # ------------------------------------------------------------------

    #: Error prefixes that indicate an ENVIRONMENTAL failure rather than a
    #: defect in the synthesized pipeline (generic runtime semantics only).
    _EXTRINSIC_ERROR_MARKERS: Tuple[str, ...] = (
        "ModuleNotFoundError",
        "ImportError",
        "FileNotFoundError",
        "FileExistsError",
        "PermissionError",
        "No such file or directory",
    )

    @staticmethod
    def _verify_egress_paths(egress_paths: Optional[List[str]]) -> Dict[str, str]:
        """Verifies declared egress artifacts exist and are non-empty."""
        verification: Dict[str, str] = {}
        for raw_path in egress_paths or []:
            path = str(raw_path)
            if not path.strip():
                continue
            if not os.path.exists(path):
                verification[path] = "missing"
            elif os.path.getsize(path) <= 0:
                verification[path] = "empty"
            else:
                verification[path] = "ok"
        return verification

    @staticmethod
    def _evaluate_verification_spec(
        verification_spec: Any,
        results: Dict[str, Any],
    ) -> List[str]:
        """
        Evaluates a pluggable verification protocol against the extracted
        namespace. Only STRUCTURAL checks are interpreted here:
          - variable identity contracts (expected_var vs actual_var resolved
            through the actual computed values),
          - registered generic postconditions (ndim / has_nans / is_deduped)
            on named result variables.
        Domain-specific rule evaluation stays in caller-side registries.
        """
        violations: List[str] = []
        if verification_spec is None:
            return violations

        checks: List[Dict[str, Any]] = []
        if isinstance(verification_spec, dict):
            checks = list(verification_spec.get("terminal_checks") or [])
        else:
            checks = list(getattr(verification_spec, "terminal_checks", None) or [])

        for check in checks:
            if not isinstance(check, dict):
                continue
            expected_var = check.get("expected_var")
            actual_var = check.get("actual_var")
            if expected_var and actual_var:
                expected_val = results.get(str(expected_var))
                actual_val = results.get(str(actual_var))
                if expected_val is not None and actual_val is not None:
                    identical = expected_val is actual_val
                    if not identical:
                        try:
                            identical = bool(expected_val == actual_val)
                        except Exception:
                            identical = False
                    if not identical:
                        violations.append(
                            f"Terminal check '{check.get('type', 'unknown')}' failed: "
                            f"'{actual_var}' does not carry the value declared for '{expected_var}'."
                        )
                continue

            prop = check.get("property") or check.get("postcondition")
            target_var = check.get("target_var") or check.get("var")
            if prop and target_var and target_var in results:
                expected = check.get("expected")
                try:
                    ok = PostconditionRegistry().evaluate(prop, results[target_var], expected)
                except Exception as eval_err:
                    logger.debug(f"[SANDBOX] postcondition '{prop}' evaluation failed: {eval_err}")
                    ok = False
                if not ok:
                    violations.append(
                        f"Postcondition '{prop}' on result '{target_var}' was not satisfied."
                    )

        return violations

    def execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        egress_paths: Optional[List[str]] = None,
        verification_spec: Optional[Any] = None,
        runtime_aliases: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute code in a separate process under the unified contract.
        Guarantees termination on timeout without leaving CPU-consuming zombie
        workers, and ALWAYS returns the full result envelope (it never raises
        for execution failures, so every caller can uniformly read 'success').
        """
        # Check if required input files exist on disk before executing
        missing_inputs: List[str] = []
        try:
            import ast
            parsed_ast = ast.parse(code)
            for node in ast.walk(parsed_ast):
                if isinstance(node, ast.Call):
                    func_name = ""
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr
                    # Skip egress/export functions (they write files, not read them)
                    if any(func_name.startswith(pfx) for pfx in ("to_", "save", "dump", "write", "export")):
                        continue
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            s = arg.value.strip()
                            _, ext = os.path.splitext(s)
                            if ext.lower() in (
                                ".csv", ".tsv", ".parquet", ".feather", ".json", ".h5", ".hdf5",
                                ".npy", ".npz", ".txt", ".png", ".jpg", ".jpeg", ".wav", ".mp3",
                            ):
                                if not os.path.exists(s) and not os.path.exists(os.path.abspath(s)):
                                    if s not in missing_inputs:
                                        missing_inputs.append(s)
        except Exception as ast_err:
            logger.debug(f"[SANDBOX] AST inspection for input files failed: {ast_err}")

        if missing_inputs:
            missing_file = missing_inputs[0]
            logger.info(f"[SANDBOX] Input file '{missing_file}' not found on disk. Skipping execution.")
            return {
                "success": False,
                "skipped": True,
                "status": "SKIPPED",
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "error": f"SKIPPED: input '{missing_file}' not provided",
                "extrinsic": False,
                "results": {},
                "egress": {},
            }

        exec_timeout = timeout if timeout is not None else self.default_timeout
        context = context or {}
        ctx = multiprocessing.get_context("spawn")
        result_queue: multiprocessing.Queue = ctx.Queue()

        worker = ctx.Process(
            target=_sandbox_worker_exec,
            args=(code, context, dict(runtime_aliases or {}), result_queue),
            daemon=True,
        )

        worker.start()

        # Drain the result queue BEFORE reaping the worker. Reading with a
        # deadline avoids the multiprocessing feeder-thread race where the
        # child's daemon queue feeder can still be flushing when the parent
        # calls join()/empty(), which previously made results vanish and
        # surfaced as "worker terminated abruptly".
        deadline = time.monotonic() + max(float(exec_timeout), 0.05)
        payload: Optional[Dict[str, Any]] = None
        timed_out = False
        try:
            remaining = max(0.05, deadline - time.monotonic())
            payload = result_queue.get(timeout=remaining)
        except _pyqueue.Empty:
            payload = None
        except Exception as recv_err:
            logger.warning(f"[SANDBOX] Result reception failed: {recv_err}")
            payload = None

        if payload is None and worker.is_alive():
            # Hard timeout: terminate, then escalate to SIGKILL.
            logger.warning(f"Worker PID {worker.pid} timed out after {exec_timeout}s. Terminating.")
            timed_out = True
            worker.terminate()
            worker.join(timeout=0.5)

            # Force kill if still unresponsive
            if worker.is_alive():
                logger.critical(f"Worker PID {worker.pid} refused SIGTERM. Issuing SIGKILL.")
                try:
                    if hasattr(signal, "SIGKILL"):
                        os.kill(worker.pid, signal.SIGKILL)
                    else:
                        worker.kill()
                except ProcessLookupError:
                    pass
                worker.join()

        if payload is not None:
            # Result delivered; reap the worker with a short grace period.
            worker.join(timeout=1.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=0.5)

        envelope: Dict[str, Any] = {
            "success": False,
            "returncode": worker.exitcode if worker.exitcode is not None else 1,
            "stdout": "",
            "stderr": "",
            "error": "",
            "extrinsic": False,
            "results": {},
            "egress": {},
        }

        if timed_out:
            envelope["error"] = f"Execution timed out after {exec_timeout} seconds."
            envelope["returncode"] = worker.exitcode if worker.exitcode is not None else 1
            return envelope

        if payload is None:
            envelope["error"] = "Execution worker terminated abruptly without returning a result."
            envelope["extrinsic"] = True
            return envelope

        envelope["success"] = bool(payload.get("success", False))
        envelope["returncode"] = int(payload.get("returncode", 0) or 0)
        envelope["stdout"] = str(payload.get("stdout", "") or "")
        envelope["stderr"] = str(payload.get("stderr", "") or "")
        envelope["error"] = str(payload.get("error", "") or "")
        envelope["results"] = dict(payload.get("results", {}) or {})

        if envelope["error"]:
            envelope["extrinsic"] = any(
                marker in envelope["error"] for marker in self._EXTRINSIC_ERROR_MARKERS
            )

        # Egress artifact verification (files persist in the shared filesystem)
        egress_verification = self._verify_egress_paths(egress_paths)
        envelope["egress"] = egress_verification
        if egress_verification:
            bad = {p: v for p, v in egress_verification.items() if v != "ok"}
            if bad:
                envelope["success"] = False
                detail = "; ".join(f"{p}: {v}" for p, v in sorted(bad.items()))
                envelope["error"] = (
                    (envelope["error"] + "\n" if envelope["error"] else "")
                    + f"Egress verification failed: {detail}"
                )
                envelope["extrinsic"] = True

        # Structural verification protocol
        spec_violations = self._evaluate_verification_spec(verification_spec, envelope["results"])
        if spec_violations:
            envelope["success"] = False
            envelope["error"] = (
                (envelope["error"] + "\n" if envelope["error"] else "")
                + "Verification contract violated: " + " | ".join(spec_violations)
            )

        return envelope

    def verify_postcondition(self, property_name: str, target: Any, expected: Any) -> bool:
        """Evaluate postconditions using the decoupled registry."""
        return self.postconditions.evaluate(property_name, target, expected)

    def verify_intent(
        self, intent_type: str, output: Any, context: Optional[Dict[str, Any]] = None
    ) -> IntentVerificationResult:
        """Evaluate terminal intents using the decoupled registry."""
        return self.intents.verify(intent_type, output, context or {})

    def is_fitted(self, estimator: Any) -> bool:
        """Check estimator readiness using the decoupled state protocol registry."""
        return self.state_protocols.is_fitted_or_ready(estimator)
