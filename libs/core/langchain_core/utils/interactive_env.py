"""Utilities for working with interactive environments."""

import sys


def is_interactive_env() -> bool:
    """Determine if running within IPython or Jupyter."""
    return hasattr(sys, "ps2")
