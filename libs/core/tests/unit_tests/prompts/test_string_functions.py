"""Test functions from langchain_core.prompts.string module."""

import pytest

from langchain_core.prompts.string import (
    get_template_variables,
    mustache_template_vars,
    jinja2_formatter,
    validate_jinja2,
    check_valid_template,
)


def test_get_template_variables_fstring() -> None:
    """Test get_template_variables with f-string format."""
    template = "Hello {name}, your role is {role}"
    result = get_template_variables(template, "f-string")
    assert result == ["name", "role"]


def test_get_template_variables_mustache_simple() -> None:
    """Test get_template_variables with mustache format - simple variables."""
    template = "Hello {{name}}, your role is {{role}}"
    result = get_template_variables(template, "mustache")
    assert result == ["name", "role"]


def test_get_template_variables_mustache_nested() -> None:
    """Test get_template_variables with mustache format - nested variables."""
    template = "Hello {{user.name}}, your role is {{user.role}}"
    result = get_template_variables(template, "mustache")
    # This should return the full nested paths, not just the top-level keys
    assert result == ["user.name", "user.role"]


def test_get_template_variables_mustache_complex_nested() -> None:
    """Test get_template_variables with mustache format - complex nested variables."""
    template = "Hello {{user.profile.name}}, your role is {{user.profile.role}} and department is {{user.department}}"
    result = get_template_variables(template, "mustache")
    assert result == ["user.profile.name", "user.profile.role", "user.department"]


def test_get_template_variables_mustache_with_sections() -> None:
    """Test get_template_variables with mustache format - variables in sections."""
    template = "{{#user}}Hello {{name}}, your role is {{role}}{{/user}}"
    result = get_template_variables(template, "mustache")
    # Variables inside sections should still be extracted
    assert result == ["user", "name", "role"]


def test_get_template_variables_mustache_with_dot_variable() -> None:
    """Test get_template_variables with mustache format - dot variable."""
    template = "Hello {{.}}, your data is {{user.name}}"
    result = get_template_variables(template, "mustache")
    # Dot variable should be ignored, nested variables should be included
    assert result == ["user.name"]


def test_get_template_variables_mustache_with_unescaped() -> None:
    """Test get_template_variables with mustache format - unescaped variables."""
    template = "Hello {{name}}, your role is {{&role}}"
    result = get_template_variables(template, "mustache")
    assert result == ["name", "role"]


def test_get_template_variables_mustache_with_inverted_sections() -> None:
    """Test get_template_variables with mustache format - inverted sections."""
    template = "{{^user}}No user{{/user}}{{#user}}Hello {{name}}{{/user}}"
    result = get_template_variables(template, "mustache")
    assert result == ["user", "name"]


@pytest.mark.requires("jinja2")
def test_get_template_variables_jinja2() -> None:
    """Test get_template_variables with jinja2 format."""
    template = "Hello {{ name }}, your role is {{ role }}"
    result = get_template_variables(template, "jinja2")
    assert result == ["name", "role"]


@pytest.mark.requires("jinja2")
def test_get_template_variables_jinja2_nested() -> None:
    """Test get_template_variables with jinja2 format - nested variables."""
    template = "Hello {{ user.name }}, your role is {{ user.role }}"
    result = get_template_variables(template, "jinja2")
    assert result == ["user.name", "user.role"]


def test_get_template_variables_invalid_format() -> None:
    """Test get_template_variables with invalid format raises ValueError."""
    template = "Hello {{name}}"
    with pytest.raises(ValueError, match="Unsupported template format"):
        get_template_variables(template, "invalid_format")


def test_mustache_template_vars_nested() -> None:
    """Test mustache_template_vars function directly with nested variables."""
    template = "Hello {{user.name}}, your role is {{user.role}}"
    result = mustache_template_vars(template)
    # This should return the full nested paths
    assert result == {"user.name", "user.role"}


def test_mustache_template_vars_simple() -> None:
    """Test mustache_template_vars function directly with simple variables."""
    template = "Hello {{name}}, your role is {{role}}"
    result = mustache_template_vars(template)
    assert result == {"name", "role"}


def test_mustache_template_vars_with_sections() -> None:
    """Test mustache_template_vars function directly with sections."""
    template = "{{#user}}Hello {{name}}, your role is {{role}}{{/user}}"
    result = mustache_template_vars(template)
    assert result == {"user", "name", "role"}


def test_mustache_template_vars_with_dot() -> None:
    """Test mustache_template_vars function directly with dot variable."""
    template = "Hello {{.}}, your data is {{user.name}}"
    result = mustache_template_vars(template)
    # Dot variable should be ignored
    assert result == {"user.name"}


def test_mustache_template_vars_with_unescaped() -> None:
    """Test mustache_template_vars function directly with unescaped variables."""
    template = "Hello {{name}}, your role is {{&role}}"
    result = mustache_template_vars(template)
    assert result == {"name", "role"}


def test_mustache_template_vars_with_inverted_sections() -> None:
    """Test mustache_template_vars function directly with inverted sections."""
    template = "{{^user}}No user{{/user}}{{#user}}Hello {{name}}{{/user}}"
    result = mustache_template_vars(template)
    assert result == {"user", "name"}


def test_mustache_template_vars_complex_nested() -> None:
    """Test mustache_template_vars function directly with complex nested variables."""
    template = "Hello {{user.profile.name}}, your role is {{user.profile.role}} and department is {{user.department}}"
    result = mustache_template_vars(template)
    assert result == {"user.profile.name", "user.profile.role", "user.department"}


def test_mustache_template_vars_deeply_nested() -> None:
    """Test mustache_template_vars function directly with deeply nested variables."""
    template = "Hello {{user.profile.personal.name}}, your role is {{user.profile.work.role}}"
    result = mustache_template_vars(template)
    assert result == {"user.profile.personal.name", "user.profile.work.role"}


def test_mustache_template_vars_mixed_nesting() -> None:
    """Test mustache_template_vars function directly with mixed nesting levels."""
    template = "Hello {{name}}, your role is {{user.role}} and department is {{user.department.info}}"
    result = mustache_template_vars(template)
    assert result == {"name", "user.role", "user.department.info"}


def test_mustache_template_vars_with_repeated_variables() -> None:
    """Test mustache_template_vars function directly with repeated variables."""
    template = "Hello {{user.name}}, your role is {{user.role}} and your name again is {{user.name}}"
    result = mustache_template_vars(template)
    # Should deduplicate variables
    assert result == {"user.name", "user.role"}


def test_mustache_template_vars_edge_cases() -> None:
    """Test mustache_template_vars function directly with edge cases."""
    # Empty template
    assert mustache_template_vars("") == set()
    
    # Template with no variables
    assert mustache_template_vars("Hello world") == set()
    
    # Template with only dot variable
    assert mustache_template_vars("{{.}}") == set()
    
    # Template with malformed variables (should still extract what it can)
    template = "Hello {{user.name}}, your role is {{user.role}} and {{malformed"
    result = mustache_template_vars(template)
    assert result == {"user.name", "user.role"} 