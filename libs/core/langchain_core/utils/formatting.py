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
        self, format_string: str, args: Sequence[Any], kwargs: Mapping[str, Any]
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
                in input_variables.
            IndexError: If the format string contains positional placeholders.

        Example:
            >>> fmt = StrictFormatter()
            >>> fmt.validate_input_variables("Hello, {name}!", ["name"])  # OK
            >>> fmt.validate_input_variables("Hello, {name}!", ["other"])  # Raises
        """
        names = set(input_variables)
        self._validate_format_string(format_string, names)

    def _validate_format_string(self, format_string: str, names: set[str]) -> None:
        """Check that every placeholder in a format string is a known variable.

        Only variable *names* are checked — placeholders are never rendered,
        so format specs (e.g. `{score:.1f}`) cannot fail validation.

        Args:
            format_string: A string containing replacement fields to validate.
            names: Valid variable names.
        """
        for _, field_name, format_spec, _ in self.parse(format_string):
            if field_name is not None:
                if not field_name:
                    msg = (
                        "Positional arguments are not allowed, everything "
                        "should be passed as keyword arguments."
                    )
                    raise IndexError(msg)
                root = field_name.split(".")[0].split("[")[0]
                if root not in names:
                    raise KeyError(root)
            if format_spec:
                self._validate_format_string(format_spec, names)


#: Default StrictFormatter instance for use throughout LangChain.
#: Used internally for formatting prompt templates with named variables.
formatter = StrictFormatter()
