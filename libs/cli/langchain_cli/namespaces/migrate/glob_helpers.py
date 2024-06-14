# Adapted from bump-pydantic
# https://github.com/pydantic/bump-pydantic
import fnmatch
import re
from pathlib import Path
from typing import List

MATCH_SEP = r"(?:/|\\)"
MATCH_SEP_OR_END = r"(?:/|\\|\Z)"
MATCH_NON_RECURSIVE = r"[^/\\]*"
MATCH_RECURSIVE = r"(?:.*)"


def glob_to_re(pattern: str) -> str:
    """Translate a glob pattern to a regular expression for matching."""
    fragments: List[str] = []
    for segment in re.split(r"/|\\", pattern):
        if segment == "":
            continue
        if segment == "**":
            # Remove previous separator match, so the recursive match c
            # can match zero or more segments.
            if fragments and fragments[-1] == MATCH_SEP:
                fragments.pop()
            fragments.append(MATCH_RECURSIVE)
        elif "**" in segment:
            raise ValueError(
                "invalid pattern: '**' can only be an entire path component"
            )
        else:
            fragment = fnmatch.translate(segment)
            fragment = fragment.replace(r"(?s:", r"(?:")
            fragment = fragment.replace(r".*", MATCH_NON_RECURSIVE)
            fragment = fragment.replace(r"\Z", r"")
            fragments.append(fragment)
        fragments.append(MATCH_SEP)
    # Remove trailing MATCH_SEP, so it can be replaced with MATCH_SEP_OR_END.
    if fragments and fragments[-1] == MATCH_SEP:
        fragments.pop()
    fragments.append(MATCH_SEP_OR_END)
    return rf"(?s:{''.join(fragments)})"


def match_glob(path: Path, pattern: str) -> bool:
    """Check if a path matches a glob pattern.

    If the pattern ends with a directory separator, the path must be a directory.
    """
    match = bool(re.fullmatch(glob_to_re(pattern), str(path)))
    if pattern.endswith("/") or pattern.endswith("\\"):
        return match and path.is_dir()
    return match
