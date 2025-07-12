"""Test mustache_template_vars and get_template_variables functions."""

import pytest

from langchain_core.prompts.string import mustache_template_vars
from langchain_core.utils.mustache import ChevronError


def test_mustache_template_vars_simple() -> None:
    template = "Hello {{name}}, your role is {{role}}"
    result = mustache_template_vars(template)
    assert result == {"name", "role"}


def test_mustache_template_vars_nested() -> None:
    template = "Hello {{user.name}}, your role is {{user.role}}"
    result = mustache_template_vars(template)
    assert result == {"user.name", "user.role"}


def test_mustache_template_vars_complex_nested() -> None:
    template = "Hello {{user.profile.name}}, {{user.profile.role}}, {{user.department}}"
    result = mustache_template_vars(template)
    assert result == {"user.profile.name", "user.profile.role", "user.department"}


def test_mustache_template_vars_with_dot_and_unescaped() -> None:
    template = "Hello {{.}}, raw role: {{&role}}, escaped: {{name}}"
    result = mustache_template_vars(template)
    assert result == {"role", "name"}


def test_mustache_template_vars_mixed_nesting() -> None:
    template = "Hello {{name}}, role: {{user.role}}, dept: {{user.department.info}}"
    result = mustache_template_vars(template)
    assert result == {"name", "user.role", "user.department.info"}


def test_mustache_template_vars_repeated() -> None:
    template = "Hello {{user.name}}, again {{user.name}}, role {{user.role}}"
    result = mustache_template_vars(template)
    assert result == {"user.name", "user.role"}


def test_mustache_template_vars_edge_cases() -> None:
    assert mustache_template_vars("") == set()
    assert mustache_template_vars("No vars") == set()
    assert mustache_template_vars("{{.}}") == set()


def test_mustache_template_vars_invalid_template() -> None:
    with pytest.raises(ChevronError):
        mustache_template_vars("Hello {{user.name}, your role is {{role}")
