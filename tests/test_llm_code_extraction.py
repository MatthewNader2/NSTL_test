"""
tests/test_llm_code_extraction.py
Tests for extract_code_from_llm_response utility across fence variants.
"""

import ast
import pytest
from src.utils import extract_code_from_llm_response


@pytest.mark.parametrize("fenced_input, expected_code", [
    # 1. Standard python fence
    (
        "```python\nimport pandas as pd\ndf = pd.read_csv('test.csv')\nout = df.dropna()\n```",
        "import pandas as pd\ndf = pd.read_csv('test.csv')\nout = df.dropna()"
    ),
    # 2. Short py fence
    (
        "```py\nresult = [x * 2 for x in range(10)]\n```",
        "result = [x * 2 for x in range(10)]"
    ),
    # 3. No language tag fence
    (
        "```\na = 10\nb = 20\nc = a + b\n```",
        "a = 10\nb = 20\nc = a + b"
    ),
    # 4. Fenced code with leading and trailing prose
    (
        "Here is the corrected code that fixes the issue:\n\n```python\nimport numpy as np\narr = np.zeros((5, 5))\nout = arr.mean()\n```\n\nHope this solves your problem!",
        "import numpy as np\narr = np.zeros((5, 5))\nout = arr.mean()"
    ),
    # 5. Plain code without fences (should be returned unchanged / stripped)
    (
        "import cv2\nimg = cv2.imread('input.png')\ngray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
        "import cv2\nimg = cv2.imread('input.png')\ngray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)"
    ),
    # 6. Single-line expression without fences
    (
        "total = sum(values)",
        "total = sum(values)"
    ),
    # 7. Fence with whitespace after lang tag
    (
        "```python   \nx = 42\ny = x ** 2\n```",
        "x = 42\ny = x ** 2"
    ),
    # 8. Empty input
    (
        "",
        ""
    )
])
def test_extract_code_from_llm_response_variants(fenced_input: str, expected_code: str):
    extracted = extract_code_from_llm_response(fenced_input)
    assert extracted == expected_code
    
    # If the expected code is non-empty, assert that it parses as valid Python AST
    if expected_code:
        try:
            parsed = ast.parse(extracted)
            assert parsed is not None
        except SyntaxError as e:
            pytest.fail(f"Extracted code failed to parse: {e}\nCode was:\n{extracted}")


def test_extract_code_preserves_indentation():
    """Verifies that internal function or loop indentation is perfectly preserved."""
    snippet = "```python\ndef compute_stats(data):\n    mean_val = sum(data) / len(data)\n    return mean_val\n```"
    extracted = extract_code_from_llm_response(snippet)
    expected = "def compute_stats(data):\n    mean_val = sum(data) / len(data)\n    return mean_val"
    assert extracted == expected
    ast.parse(extracted)


def test_plain_code_no_fences_unchanged():
    """Verifies that code with no markdown fences is returned trimmed without alteration."""
    raw_code = "import math\n\ndef circle_area(radius):\n    return math.pi * (radius ** 2)\n"
    extracted = extract_code_from_llm_response(raw_code)
    assert extracted == raw_code.strip()
    ast.parse(extracted)
