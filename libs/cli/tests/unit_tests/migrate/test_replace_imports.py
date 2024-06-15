# ruff: noqa: E402
import pytest

pytest.importorskip("libcst")


from libcst.codemod import CodemodTest

from langchain_cli.namespaces.migrate.codemods.replace_imports import (
    generate_import_replacer,
)

ReplaceImportsCodemod = generate_import_replacer(
    [
        "langchain_to_community",
        "community_to_partner",
        "langchain_to_core",
        "community_to_core",
    ]
)  # type: ignore[attr-defined]


class TestReplaceImportsCommand(CodemodTest):
    TRANSFORM = ReplaceImportsCodemod

    def test_single_import(self) -> None:
        before = """
        from langchain.chat_models import ChatOpenAI
        """
        after = """
        from langchain_community.chat_models import ChatOpenAI
        """
        self.assertCodemod(before, after)

    def test_from_community_to_partner(self) -> None:
        """Test that we can replace imports from community to partner."""
        before = """
        from langchain_community.chat_models import ChatOpenAI
        """
        after = """
        from langchain_openai import ChatOpenAI
        """
        self.assertCodemod(before, after)

    def test_noop_import(self) -> None:
        code = """
        from foo import ChatOpenAI
        """
        self.assertCodemod(code, code)

    def test_mixed_imports(self) -> None:
        before = """
        from langchain_community.chat_models import ChatOpenAI, ChatAnthropic, foo
        """
        after = """
        from langchain_community.chat_models import foo
        from langchain_anthropic import ChatAnthropic
        from langchain_openai import ChatOpenAI
        """
        self.assertCodemod(before, after)
