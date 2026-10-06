"""Signature inspection that does not evaluate annotations."""

import inspect
import sys
from collections.abc import Callable, Mapping
from typing import Any

if sys.version_info >= (3, 14):
    from annotationlib import Format


def _signature_parameters(obj: Callable[..., Any]) -> Mapping[str, inspect.Parameter]:
    """Get the parameters of a callable without evaluating its annotations.

    On Python 3.14+, annotations are evaluated lazily, so `inspect.signature` raises
    `NameError` for annotations that reference names imported only under
    `TYPE_CHECKING`. Use this when only parameter names or kinds are needed.

    Args:
        obj: The callable to inspect.

    Returns:
        The callable's parameters, keyed by name.
    """
    if sys.version_info >= (3, 14):
        signature = inspect.signature(obj, annotation_format=Format.FORWARDREF)
    else:
        signature = inspect.signature(obj)
    return signature.parameters
