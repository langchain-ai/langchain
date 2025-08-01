"""Context-Free Grammar validation for custom tools."""

from typing import Any, Optional

try:
    from lark import Lark, LarkError
except ImportError:
    Lark = None  # type: ignore[misc,assignment]
    LarkError = Exception  # type: ignore[misc,assignment]


class CFGValidator:
    """Validates input text against a Context-Free Grammar using Lark parser.

    This class provides grammar-constrained validation for custom tool outputs,
    ensuring model outputs conform to specific syntax rules while maintaining
    the flexibility of plaintext (non-JSON) outputs.
    """

    def __init__(self, grammar: str) -> None:
        """Initialize the CFG validator.

        Args:
            grammar: The Lark grammar string defining the allowed syntax.

        Raises:
            ImportError: If lark package is not installed.
            LarkError: If the grammar string is invalid.
        """
        if Lark is None:
            raise ImportError(
                "The 'lark' package is required for CFG validation. "
                "Install it with: pip install lark"
            )

        self.grammar = grammar
        try:
            self.parser = Lark(grammar, start="start")
        except Exception as e:
            raise LarkError(f"Invalid grammar definition: {e}") from e

    def validate(self, text: str) -> bool:
        """Validate input text against the grammar.

        Args:
            text: The text to validate.

        Returns:
            True if the text matches the grammar, False otherwise.
        """
        try:
            self.parser.parse(text)
            return True
        except Exception:
            return False

    def parse(self, text: str) -> Any:
        """Parse input text according to the grammar.

        Args:
            text: The text to parse.

        Returns:
            The parse tree if successful.

        Raises:
            LarkError: If the text doesn't match the grammar.
        """
        try:
            return self.parser.parse(text)
        except Exception as e:
            raise LarkError(f"Grammar validation failed: {e}") from e


def validate_cfg_format(tool_format: dict[str, Any]) -> Optional[CFGValidator]:
    """Validate and create a CFG validator from tool format specification.

    Args:
        tool_format: The format specification dictionary containing grammar rules.

    Returns:
        CFGValidator instance if format type is 'grammar', None otherwise.

    Raises:
        ValueError: If format type is 'grammar' but grammar is missing or invalid.
    """
    if not isinstance(tool_format, dict):
        return None

    if tool_format.get("type") != "grammar":
        return None

    grammar = tool_format.get("grammar")
    if not grammar:
        raise ValueError("Grammar format requires 'grammar' field")

    if not isinstance(grammar, str):
        raise ValueError("Grammar must be a string")

    try:
        return CFGValidator(grammar)
    except (ImportError, LarkError) as e:
        raise ValueError(f"Invalid grammar specification: {e}") from e


def validate_custom_tool_output(
    output: str, cfg_validator: Optional[CFGValidator]
) -> bool:
    """Validate custom tool output against CFG if validator is provided.

    Args:
        output: The tool output text to validate.
        cfg_validator: Optional CFG validator instance.

    Returns:
        True if validation passes or no validator is provided, False otherwise.
    """
    if cfg_validator is None:
        return True

    return cfg_validator.validate(output)
