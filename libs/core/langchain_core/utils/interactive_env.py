"""Utilities for working with interactive environments."""

import sys


def is_interactive_env() -> bool:
    """Determine if running within IPython or Jupyter.

    Returns:
        `True` if running in an interactive environment, `False` otherwise.
    """
    return hasattr(sys, "ps2")
