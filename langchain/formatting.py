"""Utilities for formatting strings."""
from string import Formatter
from typing import Any, Mapping, Sequence, Union
from mako.template import Template


class StrictFormatter(Formatter):
    """A subclass of formatter that checks for extra keys."""

    def check_unused_args(
        self,
        used_args: Sequence[Union[int, str]],
        args: Sequence,
        kwargs: Mapping[str, Any],
    ) -> None:
        """Check to see if extra parameters are passed."""
        extra = set(kwargs).difference(used_args)
        if extra:
            raise KeyError(extra)

    def vformat(
        self, format_string: str, args: Sequence, kwargs: Mapping[str, Any]
    ) -> str:
        """Check that no arguments are provided."""
        if len(args) > 0:
            raise ValueError(
                "No arguments should be provided, "
                "everything should be passed as keyword arguments."
            )
        return super().vformat(format_string, args, kwargs)

    def mako_format(self, format_string: str, **kwargs: Any) -> str:
        """Format a string using mako."""
        template = Template(format_string)
        return template.render(**kwargs)


formatter = StrictFormatter()
