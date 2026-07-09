"""Standalone worker for the Python grep fallback in `file_search`.

`FilesystemFileSearchMiddleware._python_search` executes this module in a
subprocess so the search is bounded by a wall-clock timeout. A user-controlled
regex can trigger catastrophic backtracking (ReDoS) in `re`, which cannot be
interrupted in-process; running the search in a child process lets the parent
kill it when the timeout expires — mirroring how `_ripgrep_search` already
bounds `rg`.

Because the child runs this file as a plain script (not as a package import),
this module must only import from the standard library and must not use
relative imports. Keeping it free of `langchain` imports also keeps the
per-search interpreter startup cheap.

Protocol: the parent writes one JSON object to stdin with keys `pattern`,
`base_path`, `root_path`, `include`, and `max_file_size_bytes`; the worker
writes a JSON object mapping virtual paths to `[line_number, line_text]`
pairs to stdout.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import sys
from pathlib import Path


def _is_within_root(candidate: Path, root: Path) -> bool:
    """Return True iff `candidate` resolves to a path inside `root` (symlinks resolved).

    Args:
        candidate: The path to check. It is resolved (following symlinks and
            normalizing `..`) before the containment check.
        root: The allowed root directory. Also resolved before comparison.

    Returns:
        `True` if the fully resolved `candidate` lies inside the resolved `root`,
        using path-segment boundaries; `False` otherwise (including on resolution
        errors).
    """
    try:
        return candidate.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def _expand_include_patterns(pattern: str) -> list[str] | None:
    """Expand brace patterns like `*.{py,pyi}` into a list of globs."""
    if "}" in pattern and "{" not in pattern:
        return None

    expanded: list[str] = []

    def _expand(current: str) -> None:
        start = current.find("{")
        if start == -1:
            expanded.append(current)
            return

        end = current.find("}", start)
        if end == -1:
            raise ValueError

        prefix = current[:start]
        suffix = current[end + 1 :]
        inner = current[start + 1 : end]
        if not inner:
            raise ValueError

        for option in inner.split(","):
            _expand(prefix + option + suffix)

    try:
        _expand(pattern)
    except ValueError:
        return None

    return expanded


def _match_include_pattern(basename: str, pattern: str) -> bool:
    """Return True if the basename matches the include pattern."""
    expanded = _expand_include_patterns(pattern)
    if not expanded:
        return False

    return any(fnmatch.fnmatch(basename, candidate) for candidate in expanded)


def _search_files(
    *,
    pattern: str,
    base_path: str,
    root_path: str,
    include: str | None,
    max_file_size_bytes: int,
) -> dict[str, list[tuple[int, str]]]:
    """Search files under `base_path` for regex matches, line by line.

    Args:
        pattern: Regular expression applied to each line.
        base_path: Directory to walk. Already validated by the parent to lie
            inside `root_path`.
        root_path: Resolved search root, used for containment checks and to
            build virtual paths.
        include: Optional glob filter applied to file basenames.
        max_file_size_bytes: Files larger than this are skipped.

    Returns:
        Mapping of virtual file path to `(line_number, line_text)` matches.
    """
    regex = re.compile(pattern)
    root = Path(root_path)
    results: dict[str, list[tuple[int, str]]] = {}

    # Walk directory tree without following symlinked directories so traversal
    # cannot leave the root via a symlinked subdirectory.
    for walk_root, _dirs, files in os.walk(base_path, followlinks=False):
        for name in files:
            file_path = Path(walk_root) / name

            # Re-check containment after resolving so an in-root symlinked file
            # pointing outside the root is never read.
            if not _is_within_root(file_path, root):
                continue

            if not file_path.is_file():
                continue

            # Check include filter
            if include and not _match_include_pattern(file_path.name, include):
                continue

            # Skip files that are too large, and files that disappear or become
            # inaccessible mid-walk.
            try:
                if file_path.stat().st_size > max_file_size_bytes:
                    continue
            except OSError:
                continue

            try:
                content = file_path.read_text()
            except (UnicodeDecodeError, PermissionError):
                continue

            # Search content
            for line_num, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    virtual_path = "/" + str(file_path.relative_to(root))
                    if virtual_path not in results:
                        results[virtual_path] = []
                    results[virtual_path].append((line_num, line))

    return results


def main() -> None:
    """Run one search request read from stdin and write results to stdout."""
    request = json.load(sys.stdin)
    results = _search_files(
        pattern=request["pattern"],
        base_path=request["base_path"],
        root_path=request["root_path"],
        include=request["include"],
        max_file_size_bytes=request["max_file_size_bytes"],
    )
    # Default `ensure_ascii` keeps the payload pure ASCII, so no stream
    # encoding configuration is needed on either side of the pipe.
    json.dump(results, sys.stdout)


if __name__ == "__main__":
    main()
