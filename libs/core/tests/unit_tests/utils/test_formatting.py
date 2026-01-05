"""Tests for langchain_core.utils.formatting."""

import pytest

from langchain_core.utils.formatting import StrictFormatter, formatter


class TestStrictFormatter:
    """Tests for the `StrictFormatter` class."""

    def test_vformat_with_keyword_args(self) -> None:
        """Test that vformat works with keyword arguments."""
        fmt = StrictFormatter()
        result = fmt.vformat("Hello, {name}!", [], {"name": "World"})
        assert result == "Hello, World!"

    def test_vformat_with_multiple_keyword_args(self) -> None:
        """Test vformat with multiple keyword arguments."""
        fmt = StrictFormatter()
        result = fmt.vformat(
            "{greeting}, {name}! You have {count} messages.",
            [],
            {"greeting": "Hello", "name": "Alice", "count": 5},
        )
        assert result == "Hello, Alice! You have 5 messages."

    def test_vformat_with_empty_string(self) -> None:
        """Test `vformat` with empty format string."""
        fmt = StrictFormatter()
        result = fmt.vformat("", [], {})
        assert result == ""

    def test_vformat_with_no_placeholders(self) -> None:
        """Test `vformat` with no placeholders in format string."""
        fmt = StrictFormatter()
        result = fmt.vformat("Hello, World!", [], {})
        assert result == "Hello, World!"

    def test_vformat_raises_on_positional_args(self) -> None:
        """Test that `vformat` raises `ValueError` when positional args are provided."""
        fmt = StrictFormatter()
        with pytest.raises(
            ValueError,
            match=r"No arguments should be provided, "
            r"everything should be passed as keyword arguments\.",
        ):
            fmt.vformat("{}", ["arg"], {})

    def test_vformat_raises_on_multiple_positional_args(self) -> None:
        """Test that `vformat` raises `ValueError` with multiple positional args."""
        fmt = StrictFormatter()
        with pytest.raises(ValueError, match=r"No arguments should be provided"):
            fmt.vformat("{} {}", ["arg1", "arg2"], {})

    def test_vformat_with_special_characters(self) -> None:
        """Test `vformat` with special characters in values."""
        fmt = StrictFormatter()
        result = fmt.vformat("{text}", [], {"text": "Hello\nWorld\t!"})
        assert result == "Hello\nWorld\t!"

    def test_vformat_with_unicode(self) -> None:
        """Test `vformat` with unicode characters."""
        fmt = StrictFormatter()
        result = fmt.vformat(
            "{emoji} {text}", [], {"emoji": "🎉", "text": "こんにちは"}
        )
        assert result == "🎉 こんにちは"

    def test_vformat_with_format_spec(self) -> None:
        """Test `vformat` with format specifications."""
        fmt = StrictFormatter()
        result = fmt.vformat("{num:.2f}", [], {"num": 3.14159})
        assert result == "3.14"

    def test_vformat_with_nested_braces(self) -> None:
        """Test `vformat` with escaped braces."""
        fmt = StrictFormatter()
        result = fmt.vformat("{{literal}} {var}", [], {"var": "value"})
        assert result == "{literal} value"

    def test_validate_input_variables_success(self) -> None:
        """Test that `validate_input_variables` succeeds with valid input."""
        fmt = StrictFormatter()
        # Should not raise
        fmt.validate_input_variables("{name} {age}", ["name", "age"])

    def test_validate_input_variables_with_extra_variables(self) -> None:
        """Test `validate_input_variables` with extra variables (should succeed)."""
        fmt = StrictFormatter()
        # Extra variables are allowed
        fmt.validate_input_variables("{name}", ["name", "extra"])

    def test_validate_input_variables_with_missing_variable(self) -> None:
        """Test `validate_input_variables` raises with missing variable."""
        fmt = StrictFormatter()
        with pytest.raises(KeyError):
            fmt.validate_input_variables("{name} {missing}", ["name"])

    def test_validate_input_variables_empty_format(self) -> None:
        """Test `validate_input_variables` with empty format string."""
        fmt = StrictFormatter()
        # Should not raise
        fmt.validate_input_variables("", [])

    def test_validate_input_variables_no_placeholders(self) -> None:
        """Test `validate_input_variables` with no placeholders."""
        fmt = StrictFormatter()
        # Should not raise
        fmt.validate_input_variables("Hello, World!", [])


class TestFormatterSingleton:
    """Tests for the formatter singleton instance."""

    def test_formatter_is_strict_formatter(self) -> None:
        """Test that the formatter singleton is a `StrictFormatter` instance."""
        assert isinstance(formatter, StrictFormatter)

    def test_formatter_format_works(self) -> None:
        """Test that the formatter singleton can format strings."""
        result = formatter.format("{greeting}, {name}!", greeting="Hello", name="World")
        assert result == "Hello, World!"

    def test_formatter_rejects_positional_args(self) -> None:
        """Test that the formatter singleton rejects positional arguments."""
        with pytest.raises(ValueError, match=r"No arguments should be provided"):
            formatter.format("{}", "arg")
