"""Utilities for formatting strings."""

from collections.abc import Mapping, Sequence
from string import Formatter
from typing import Any


class StrictFormatter(Formatter):
    """Formatter that checks for extra keys."""

    def vformat(
        self, format_string: str, args: Sequence, kwargs: Mapping[str, Any]
    ) -> str:
        """Check that no arguments are provided.

        Args:
            format_string: The format string.
            args: The arguments.
            kwargs: The keyword arguments.

        Returns:
            The formatted string.

        Raises:
            ValueError: If any arguments are provided.
        """
        if len(args) > 0:
            raise ValueError(
                "No arguments should be provided, "
                "everything should be passed as keyword arguments."
            )
        return super().vformat(format_string, args, kwargs)

    def validate_input_variables(
        self, format_string: str, input_variables: list[str]
    ) -> None:
        """Check that all input variables are used in the format string.

        Args:
            format_string: The format string.
            input_variables: The input variables.

        Raises:
            ValueError: If any input variables are not used in the format string.
        """
        dummy_inputs = {input_variable: "foo" for input_variable in input_variables}
        super().format(format_string, **dummy_inputs)


formatter = StrictFormatter()
