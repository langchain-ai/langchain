"""Utilities for working with interactive environments."""


def is_interactive_env() -> bool:
    """Determine if running within IPython or Jupyter.

    Returns:
        True if running in an interactive environment, False otherwise.
    """
    import sys

    return hasattr(sys, "ps2")
