"""
src/gevr_sandbox.py - Neuro-Symbolic Topological Lattice (NSTL)
Generate-Execute-Verify-Repair (GEVR) Execution Sandbox.
Executes candidate scripts in isolated subprocesses and manages feedback repair loops.
"""

from __future__ import annotations
import ast
import os
import sys
import tempfile
import subprocess
from typing import Tuple, Optional, Callable
from log_config import get_logger

logger = get_logger('gevr_sandbox')


class GEVRSandbox:
    """
    Isolated execution environment for synthesized code verification.
    Guarantees strict timeout and process isolation.
    """
    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout = timeout_seconds

    def execute_and_verify(self, code: str) -> Tuple[bool, str, str]:
        """
        Executes Python code in an isolated subprocess.
        Returns: (success: bool, stdout: str, stderr: str)
        """
        if not code or not code.strip():
            return False, "", "ExecutionError: Empty code block."

        # Pre-execution AST syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, "", f"SyntaxError: {e}"

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f:
                f.write(code)
                temp_path = f.name

            res = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            success = (res.returncode == 0)
            return success, res.stdout, res.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"ExecutionTimedOut: Exceeded {self.timeout}s execution limit."
        except Exception as e:
            return False, "", f"ExecutionSystemError: {e}"
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

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
            success, stdout, stderr = self.execute_and_verify(current_code)
            if success:
                logger.info(f"[GEVR VERIFIED] Code executed cleanly on attempt {attempt + 1}")
                return True, current_code, stdout

            logger.warning(f"[GEVR FAILED] Attempt {attempt + 1} traceback:\n{stderr.strip()}")

            if llm_repair_func is not None and attempt + 1 < max_attempts:
                try:
                    # Provide exact code and stderr traceback to the feedback model
                    repaired_text = llm_repair_func(current_code, stderr)
                    clean_code = self._extract_clean_code(repaired_text)

                    if clean_code:
                        ast.parse(clean_code)  # Verify syntax before accepting
                        current_code = clean_code
                        logger.info(f"[GEVR REPAIR] Feedback model generated candidate fix for attempt {attempt + 2}")
                        continue
                except Exception as e:
                    logger.error(f"[GEVR REPAIR ERROR] Feedback loop failed: {e}")

            break

        return False, current_code, stderr

    @staticmethod
    def _extract_clean_code(raw_text: str) -> str:
        """Extracts pure executable Python code from LLM responses."""
        if not raw_text:
            return ""

        # Extract from markdown block if present
        if "```python" in raw_text:
            return raw_text.split("```python")[1].split("```")[0].strip()
        elif "```" in raw_text:
            return raw_text.split("```")[1].split("```")[0].strip()

        return raw_text.strip()
