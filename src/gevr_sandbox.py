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
import signal
import resource
from typing import Tuple, Optional, Callable, Dict, Any
from log_config import get_logger
from config import settings
from utils import extract_code_from_llm_response

logger = get_logger('gevr_sandbox')


def _init_worker(paths: list[str]):
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)
    for pkg in ['numpy', 'pandas', 'cv2', 'sklearn', 'matplotlib']:
        try:
            mod = __import__(pkg)
            if pkg == 'matplotlib':
                mod.use('Agg')
        except ImportError:
            pass  # Package not installed, skip pre-warming


_BLOCKED_MODULES = frozenset({
    'os', 'subprocess', 'shutil', 'socket', 'sys', 'ctypes',
    'signal', 'resource', 'importlib', 'pathlib', 'tempfile',
    'multiprocessing', 'threading', 'http', 'urllib', 'ftplib',
    'smtplib', 'telnetlib', 'xmlrpc', 'code', 'codeop', 'compile',
    'compileall', 'py_compile', 'zipimport', 'pkgutil',
})

def _restricted_import(name, *args, **kwargs):
    base = name.split('.')[0]
    if base in _BLOCKED_MODULES:
        raise ImportError(f"Import of '{name}' is blocked in NSTL sandbox for security.")
    return __builtins__.__import__(name, *args, **kwargs) if hasattr(__builtins__, '__import__') else __import__(name, *args, **kwargs)

def _sandbox_worker_exec(code: str) -> Dict[str, Any]:
    """Isolated execution unit executed within persistent worker process."""
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
            return {
                "success": True,
                "stdout": stdout_buf.getvalue(),
                "stderr": stderr_buf.getvalue(),
                "error": ""
            }
        except Exception:
            return {
                "success": False,
                "stdout": stdout_buf.getvalue(),
                "stderr": stderr_buf.getvalue(),
                "error": traceback.format_exc()
            }


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

    def execute(self, code: str, timeout: Optional[float] = None) -> Dict[str, Any]:
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
        try:
            self._ensure_pool()
            async_res = self._pool.apply_async(_sandbox_worker_exec, (code,))
            res = async_res.get(timeout=tout)
            if res.get("error") is None:
                res["error"] = ""
            return res
        except multiprocessing.TimeoutError:
            return {"success": False, "stdout": "", "stderr": "", "error": f"ExecutionTimedOut: Exceeded {tout}s execution limit."}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": "", "error": f"ExecutionSystemError: {e}"}

    def execute_and_verify(self, code: str) -> Tuple[bool, str, str]:
        """
        Executes Python code and returns (success, stdout, stderr/error).
        """
        res = self.execute(code)
        err = res.get("error", "") or res.get("stderr", "")
        return res["success"], res["stdout"], err

    def repair_cycle(
        self,
        initial_code: str,
        llm_repair_func: Optional[Callable[[str, str], str]] = None,
        max_attempts: int = 2
    ) -> Tuple[bool, str, str]:
        """
        Feedback verification loop:
        Executes code, captures tracebacks, and applies diagnostic LLM repairs.
        """
        current_code = initial_code
        for attempt in range(max_attempts):
            success, stdout, error = self.execute_and_verify(current_code)
            if success:
                logger.info(f"[GEVR Sandbox] Verification PASSED on attempt {attempt + 1}")
                return True, current_code, ""

            logger.warning(f"[GEVR Sandbox] Attempt {attempt + 1} failed with error:\n{error}")
            if attempt < max_attempts - 1 and llm_repair_func:
                logger.info(f"[GEVR Sandbox] Requesting repair heuristic for attempt {attempt + 2}...")
                repaired = extract_code_from_llm_response(llm_repair_func(current_code, error))
                if repaired and repaired != current_code:
                    current_code = repaired
                else:
                    break
            else:
                break

        return False, current_code, error
