# gevr_sandbox.py
"""
Generate-Execute-Verify-Repair (GEVR) Sandboxed Execution Engine for NSTL.
Runs candidate python code in an isolated subprocess, intercepts runtime tracebacks,
applies AST heuristic fixes, and optionally invokes LLM repair fallback.
"""

import ast
import os
import re
import sys
import tempfile
import subprocess
from typing import Tuple, Optional, Dict, Any, Callable
from log_config import get_logger

logger = get_logger('gevr_sandbox')

# Import canonical map from unification if available
try:
    from unification import CANONICAL_IMPORT_MAP
except ImportError:
    CANONICAL_IMPORT_MAP = {
        "pd": ("import pandas as pd", "pandas"),
        "np": ("import numpy as np", "numpy"),
        "plt": ("import matplotlib.pyplot as plt", "matplotlib"),
        "cv2": ("import cv2", "cv2"),
        "sk": ("import sklearn as sk", "sklearn"),
        "sp": ("import scipy as sp", "scipy"),
    }


class GEVRSandbox:
    def __init__(self, timeout_seconds: int = 5):
        self.timeout = timeout_seconds

    def execute_and_verify(self, code: str) -> Tuple[bool, str, str]:
        """
        Runs Python code in an isolated subprocess with a strict timeout.
        Returns: (success: bool, stdout: str, stderr: str)
        """
        if not code or not code.strip():
            return False, "", "ExecutionError: Empty code block."

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f:
            f.write(code)
            temp_path = f.name

        try:
            res = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            success = (res.returncode == 0)
            return success, res.stdout, res.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"ExecutionTimedOut: Exceeded {self.timeout}.0 seconds timeout."
        except Exception as e:
            return False, "", f"ExecutionSystemError: {e}"
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def repair_cycle(
        self, 
        initial_code: str, 
        llm_repair_func: Optional[Callable[[str], str]] = None, 
        max_attempts: int = 3
    ) -> Tuple[bool, str, str]:
        """
        Executes an automated feedback repair loop using AST heuristic mutation + LLM fallback.
        """
        current_code = initial_code

        for attempt in range(max_attempts):
            success, stdout, stderr = self.execute_and_verify(current_code)
            if success:
                logger.info(f"[GEVR SUCCESS] Code passed verification on attempt {attempt + 1}")
                return True, current_code, stdout

            logger.warning(f"[GEVR VERIFY FAILURE] Attempt {attempt + 1} failed with stderr:\n{stderr.strip()}")

            # Phase 1: Deterministic AST Repairs
            mutated_code = self._apply_ast_heuristic_repairs(current_code, stderr)
            if mutated_code != current_code:
                current_code = mutated_code
                logger.info(f"[GEVR HEURISTIC MUTATION] Applied AST repair for attempt {attempt + 2}")
                continue

            # Phase 2: Targeted LLM Traceback Repair (if provided)
            if llm_repair_func is not None:
                try:
                    repaired = llm_repair_func(current_code)
                    if repaired and repaired.strip():
                        clean_code = self._extract_clean_code(repaired)
                        if clean_code and clean_code != current_code:
                            try:
                                ast.parse(clean_code)
                                current_code = clean_code
                                logger.info(f"[GEVR LLM REPAIR] Applied LLM fix for attempt {attempt + 2}")
                                continue
                            except SyntaxError:
                                logger.warning("[GEVR LLM REPAIR] Rejected LLM output with syntax error.")
                except Exception as e:
                    logger.error(f"[GEVR LLM REPAIR ERROR] Failed to invoke LLM repair: {e}")

            break

        return False, current_code, stderr

    def _extract_clean_code(self, raw_text: str) -> str:
        """Extracts pure Python code block from LLM output, stripping markdown and conversational text."""
        if not raw_text:
            return ""
        code_match = re.search(r"```(?:python)?\s*\n(.*?)\n```", raw_text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # If no markdown blocks, filter out lines starting with prompt text
        lines = []
        for line in raw_text.splitlines():
            if line.strip().startswith("Fix this") or line.strip().startswith("###"):
                continue
            lines.append(line)
        return "\n".join(lines).strip()


    def _apply_ast_heuristic_repairs(self, code: str, error_trace: str) -> str:
        """Applies deterministic AST/regex fixes based on standard tracebacks."""
        # 1. Fix missing module import if reported via NameError
        name_err = re.search(r"NameError: name '([a-zA-Z0-9_]+)' is not defined", error_trace)
        if name_err:
            missing_name = name_err.group(1)
            try:
                from unification import resolve_unbound_module
                stmt = resolve_unbound_module(missing_name)
                if stmt and stmt not in code:
                    return f"{stmt}\n{code}"
            except ImportError:
                if missing_name in CANONICAL_IMPORT_MAP:
                    stmt, _ = CANONICAL_IMPORT_MAP[missing_name]
                    if stmt not in code:
                        return f"{stmt}\n{code}"

        # 2. Fix module not found if standard alias was used
        mod_err = re.search(r"ModuleNotFoundError: No module named '([a-zA-Z0-9_]+)'", error_trace)
        if mod_err:
            bad_mod = mod_err.group(1)
            # Remove bad module import line if generated
            lines = [l for l in code.splitlines() if not re.match(fr"^\s*import\s+{bad_mod}\b", l)]
            return "\n".join(lines)

        return code
