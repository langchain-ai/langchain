"""Utilities for formatting strings."""

from collections.abc import Mapping, Sequence
from string import Formatter
from typing import Any


class StrictFormatter(Formatter):
    """A string formatter that enforces keyword-only argument substitution.

    This formatter extends Python's built-in `string.Formatter` to provide stricter
    validation for prompt template formatting. It ensures that all variable
    substitutions use keyword arguments rather than positional arguments, which improves
    clarity and reduces errors when formatting prompt templates.

    Example:
        >>> fmt = StrictFormatter()
        >>> fmt.format("Hello, {name}!", name="World")
        'Hello, World!'
        >>> fmt.format("Hello, {}!", "World")  # Raises ValueError
    """

    def vformat(
        self, format_string: str, args: Sequence, kwargs: Mapping[str, Any]
    ) -> str:
        """Format a string using only keyword arguments.

        Overrides the base `vformat` to reject positional arguments, ensuring all
        substitutions are explicit and named.

        Args:
            format_string: A string containing replacement fields (e.g., `'{name}'`).
            args: Positional arguments (must be empty).
            kwargs: Keyword arguments for substitution into the format string.

        Returns:
            The formatted string with all replacement fields substituted.

        Raises:
            ValueError: If any positional arguments are provided.
        """
        if len(args) > 0:
            msg = (
                "No arguments should be provided, "
                "everything should be passed as keyword arguments."
            )
            raise ValueError(msg)
        return super().vformat(format_string, args, kwargs)

    def validate_input_variables(
        self, format_string: str, input_variables: list[str]
    ) -> None:
        """Validate that input variables match the placeholders in a format string.

        Checks that the provided input variables can be used to format the given string
        without missing or extra keys. This is useful for validating prompt templates
        before runtime.

        Args:
            format_string: A string containing replacement fields to validate
                against (e.g., `'Hello, {name}!'`).
            input_variables: List of variable names expected to fill the
                replacement fields.

        Raises:
            KeyError: If the format string contains placeholders not present
                in input_variables, or if input_variables contains extra variables
                not present in the format string.

        Example:
            >>> fmt = StrictFormatter()
            >>> fmt.validate_input_variables("Hello, {name}!", ["name"])  # OK
            >>> fmt.validate_input_variables("Hello, {name}!", ["other"])  # Raises
        """
        # Get all placeholders in the format string
        placeholders = {
            fname
            for _, fname, _, _ in self.parse(format_string)
            if fname
        }

        # Check for missing variables (placeholders not in input_variables)
        missing = placeholders - set(input_variables)
        if missing:
            raise KeyError(f"Missing variables in input_variables: {missing}")

        # Check for extra variables (input_variables not in placeholders)
        extra = set(input_variables) - placeholders
        if extra:
            raise KeyError(f"Extra variables in input_variables not used: {extra}")

        dummy_inputs = dict.fromkeys(input_variables, "foo")
        super().format(format_string, **dummy_inputs)


#: Default StrictFormatter instance for use throughout LangChain.
#: Used internally for formatting prompt templates with named variables.
formatter = StrictFormatter()
