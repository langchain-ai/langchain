"""Bug condition exploration test for type: ignore cleanup in ai.py.

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5

For each removable `type: ignore` comment in ai.py, this test strips the
comment, writes a temp shadow file, and runs mypy with ``--shadow-file`` so
the project context (imports, pydantic plugin) is preserved.

On UNFIXED code, mypy will report errors — confirming the ignores are currently
necessary. Once the fix is applied (type: ignore replaced with cast()), mypy
should pass cleanly on the fixed code as-is.
"""

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

# Path to the source file under test
AI_PY = Path(__file__).resolve().parents[2] / "langchain_core" / "messages" / "ai.py"

# Project root for running mypy with correct config
CORE_ROOT = AI_PY.resolve().parents[2]

# Patterns for the 5 originally-removable type: ignore lines.
# On UNFIXED code these lines contain both the pattern AND a type: ignore comment.
# On FIXED code the type: ignore comments have been replaced with cast() calls,
# so the original patterns may no longer appear verbatim.
_REMOVABLE_PATTERNS: list[tuple[str, str]] = [
    # (substring unique to the line, expected mypy error code)
    ('tool_call_block["index"] = tool_call["index"]', "typeddict-item"),
    ('tool_call_block["extras"] = tool_call["extras"]', "typeddict-item"),
    ('self.content[idx]["extras"] = block["extras"]', "index"),
    ('self.content[idx]["type"] = "server_tool_call"', "index"),
    ('self.content[idx]["args"] = args', "index"),
]

# Regex to match type: ignore comments (with optional error codes)
TYPE_IGNORE_RE = re.compile(r"\s*#\s*type:\s*ignore\[.*?\]")


def _find_removable_lines() -> list[tuple[int, str]]:
    """Find the actual line numbers of the removable type: ignore comments.

    Returns an empty list if the fix has already been applied (no type: ignore
    comments remain on the target lines).
    """
    source_lines = AI_PY.read_text().splitlines()
    result: list[tuple[int, str]] = []
    for pattern, error_code in _REMOVABLE_PATTERNS:
        for i, line in enumerate(source_lines, 1):
            if pattern in line and "type: ignore" in line:
                result.append((i, error_code))
                break
    return result


def _code_is_fixed() -> bool:
    """Check whether the fix has been applied.

    The fix replaces the 5 removable type: ignore comments with cast() calls.
    If none of the original patterns have type: ignore comments, the fix is applied.
    """
    return len(_find_removable_lines()) == 0


REMOVABLE_LINES = _find_removable_lines()
CODE_IS_FIXED = _code_is_fixed()


def _strip_type_ignore(source: str, target_line: int) -> str:
    """Remove the type: ignore comment from a specific line number (1-indexed)."""
    lines = source.splitlines(keepends=True)
    idx = target_line - 1
    lines[idx] = TYPE_IGNORE_RE.sub("", lines[idx])
    return "".join(lines)


def _run_mypy_on_source(source_text: str) -> subprocess.CompletedProcess[str]:
    """Run mypy on a modified version of ai.py using --shadow-file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as tmp:
        tmp.write(source_text)
        tmp.flush()
        shadow_path = tmp.name

    try:
        return subprocess.run(
            [
                "uv",
                "run",
                "--group",
                "typing",
                "mypy",
                "--no-incremental",
                "--shadow-file",
                str(AI_PY),
                shadow_path,
                "--no-error-summary",
                str(AI_PY),
            ],
            capture_output=True,
            text=True,
            cwd=str(CORE_ROOT),
        )
    finally:
        Path(shadow_path).unlink(missing_ok=True)


@pytest.mark.skipif(
    CODE_IS_FIXED,
    reason="Fix already applied — type: ignore comments removed; see test_fixed_code_passes_mypy",
)
@pytest.mark.parametrize(
    ("line_number", "expected_error_code"),
    REMOVABLE_LINES if REMOVABLE_LINES else [pytest.param(0, "", marks=pytest.mark.skip)],
    ids=[f"line_{ln}" for ln, _ in REMOVABLE_LINES] if REMOVABLE_LINES else ["skip"],
)
def test_type_ignore_removable(line_number: int, expected_error_code: str) -> None:
    """Stripping a removable type: ignore should leave mypy clean.

    On UNFIXED code this test FAILS — mypy reports errors without the ignore.
    After the fix is applied, mypy should pass (exit 0).
    """
    source = AI_PY.read_text()
    modified = _strip_type_ignore(source, line_number)
    result = _run_mypy_on_source(modified)

    # We assert mypy exits cleanly (code 0).
    # On unfixed code this will FAIL because mypy reports errors.
    assert result.returncode == 0, (
        f"mypy reported errors on line {line_number} after removing "
        f"type: ignore[{expected_error_code}]:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.skipif(
    not CODE_IS_FIXED,
    reason="Fix not yet applied — run test_type_ignore_removable instead",
)
def test_fixed_code_passes_mypy() -> None:
    """After the fix, the current ai.py should pass mypy cleanly.

    Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5

    The fix replaced the 5 removable type: ignore comments with cast() calls.
    This test verifies mypy passes on the fixed code as-is.
    """
    source = AI_PY.read_text()
    result = _run_mypy_on_source(source)

    assert result.returncode == 0, (
        f"mypy reported errors on fixed ai.py:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
