from __future__ import annotations

from pathlib import Path

import pytest

from langchain_cli.namespaces.migrate.glob_helpers import glob_to_re, match_glob


class TestGlobHelpers:
    match_glob_values: list[tuple[str, Path, bool]] = [
        ("foo", Path("foo"), True),
        ("foo", Path("bar"), False),
        ("foo", Path("foo/bar"), False),
        ("*", Path("foo"), True),
        ("*", Path("bar"), True),
        ("*", Path("foo/bar"), False),
        ("**", Path("foo"), True),
        ("**", Path("foo/bar"), True),
        ("**", Path("foo/bar/baz/qux"), True),
        ("foo/bar", Path("foo/bar"), True),
        ("foo/bar", Path("foo"), False),
        ("foo/bar", Path("far"), False),
        ("foo/bar", Path("foo/foo"), False),
        ("foo/*", Path("foo/bar"), True),
        ("foo/*", Path("foo/bar/baz"), False),
        ("foo/*", Path("foo"), False),
        ("foo/*", Path("bar"), False),
        ("foo/**", Path("foo/bar"), True),
        ("foo/**", Path("foo/bar/baz"), True),
        ("foo/**", Path("foo/bar/baz/qux"), True),
        ("foo/**", Path("foo"), True),
        ("foo/**", Path("bar"), False),
        ("foo/**/bar", Path("foo/bar"), True),
        ("foo/**/bar", Path("foo/baz/bar"), True),
        ("foo/**/bar", Path("foo/baz/qux/bar"), True),
        ("foo/**/bar", Path("foo/baz/qux"), False),
        ("foo/**/bar", Path("foo/bar/baz"), False),
        ("foo/**/bar", Path("foo/bar/bar"), True),
        ("foo/**/bar", Path("foo"), False),
        ("foo/**/bar", Path("bar"), False),
        ("foo/**/*/bar", Path("foo/bar"), False),
        ("foo/**/*/bar", Path("foo/baz/bar"), True),
        ("foo/**/*/bar", Path("foo/baz/qux/bar"), True),
        ("foo/**/*/bar", Path("foo/baz/qux"), False),
        ("foo/**/*/bar", Path("foo/bar/baz"), False),
        ("foo/**/*/bar", Path("foo/bar/bar"), True),
        ("foo/**/*/bar", Path("foo"), False),
        ("foo/**/*/bar", Path("bar"), False),
        ("foo/ba*", Path("foo/bar"), True),
        ("foo/ba*", Path("foo/baz"), True),
        ("foo/ba*", Path("foo/qux"), False),
        ("foo/ba*", Path("foo/baz/qux"), False),
        ("foo/ba*", Path("foo/bar/baz"), False),
        ("foo/ba*", Path("foo"), False),
        ("foo/ba*", Path("bar"), False),
        ("foo/**/ba*/*/qux", Path("foo/a/b/c/bar/a/qux"), True),
        ("foo/**/ba*/*/qux", Path("foo/a/b/c/baz/a/qux"), True),
        ("foo/**/ba*/*/qux", Path("foo/a/bar/a/qux"), True),
        ("foo/**/ba*/*/qux", Path("foo/baz/a/qux"), True),
        ("foo/**/ba*/*/qux", Path("foo/baz/qux"), False),
        ("foo/**/ba*/*/qux", Path("foo/a/b/c/qux/a/qux"), False),
        ("foo/**/ba*/*/qux", Path("foo"), False),
        ("foo/**/ba*/*/qux", Path("bar"), False),
    ]

    @pytest.mark.parametrize(("pattern", "path", "expected"), match_glob_values)
    def test_match_glob(self, pattern: str, path: Path, expected: bool):
        expr = glob_to_re(pattern)
        assert (
            match_glob(path, pattern) == expected
        ), f"path: {path}, pattern: {pattern}, expr: {expr}"
