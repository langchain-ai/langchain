"""Integration tests: verify the module compiles and the tool can be imported."""

from __future__ import annotations

import sys
import unittest


class TestCompile(unittest.TestCase):
    def test_import_modules(self):
        import langchain_rustchain  # noqa: F401
        from langchain_rustchain import RustChainTool  # noqa: F401

        self.assertTrue(hasattr(langchain_rustchain, "RustChainTool"))
        self.assertTrue(issubclass(RustChainTool, object))

    def test_version(self):
        from langchain_rustchain._version import __version__

        self.assertRegex(__version__, r"^\d+\.\d+\.\d+$")


if __name__ == "__main__":
    sys.exit(unittest.main())