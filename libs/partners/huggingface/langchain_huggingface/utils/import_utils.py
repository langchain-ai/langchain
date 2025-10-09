from __future__ import annotations

import importlib.metadata
import importlib.util
import operator as op

from packaging import version

STR_OPERATION_TO_FUNC = {
    ">": op.gt,
    ">=": op.ge,
    "==": op.eq,
    "!=": op.ne,
    "<=": op.le,
    "<": op.lt,
}


_optimum_available = importlib.util.find_spec("optimum") is not None
_optimum_version = "N/A"
if _optimum_available:
    try:
        _optimum_version = importlib.metadata.version("optimum")
    except importlib.metadata.PackageNotFoundError:
        _optimum_available = False


_optimum_intel_available = (
    _optimum_available and importlib.util.find_spec("optimum.intel") is not None
)
_optimum_intel_version = "N/A"
if _optimum_intel_available:
    try:
        _optimum_intel_version = importlib.metadata.version("optimum-intel")
    except importlib.metadata.PackageNotFoundError:
        _optimum_intel_available = False


_ipex_available = importlib.util.find_spec("intel_extension_for_pytorch") is not None

_openvino_available = importlib.util.find_spec("openvino") is not None


# This function was copied from: https://github.com/huggingface/accelerate/blob/874c4967d94badd24f893064cc3bef45f57cadf7/src/accelerate/utils/versions.py#L319
def compare_versions(
    library_or_version: str | version.Version,
    operation: str,
    requirement_version: str,
) -> bool:
    """Compare a library version to some requirement using a given operation.

    Args:
        library_or_version:
            A library name or a version to check.
        operation:
            A string representation of an operator, such as `">"` or `"<="`.
        requirement_version:
            The version to compare the library version against

    """
    if operation not in STR_OPERATION_TO_FUNC:
        msg = (
            f"`operation` must be one of {list(STR_OPERATION_TO_FUNC.keys())}"
            f", received {operation}"
        )
        raise ValueError(msg)
    if isinstance(library_or_version, str):
        library_or_version = version.parse(
            importlib.metadata.version(library_or_version)
        )
    return STR_OPERATION_TO_FUNC[operation](
        library_or_version, version.parse(requirement_version)
    )


def is_optimum_available() -> bool:
    return _optimum_available


def is_optimum_intel_available() -> bool:
    return _optimum_intel_available


def is_ipex_available() -> bool:
    return _ipex_available


def is_openvino_available() -> bool:
    return _openvino_available


def is_optimum_version(operation: str, reference_version: str) -> bool:
    """Compare the current Optimum version to a given reference with an operation."""
    if not _optimum_version:
        return False
    return compare_versions(
        version.parse(_optimum_version), operation, reference_version
    )


def is_optimum_intel_version(operation: str, reference_version: str) -> bool:
    """Compare current Optimum Intel version to a given reference with an operation."""
    if not _optimum_intel_version:
        return False
    return compare_versions(
        version.parse(_optimum_intel_version), operation, reference_version
    )


IMPORT_ERROR = """
requires the {0} library but it was not found in your environment.
You can install it with pip: `pip install {0}`.
Please note that you may need to restart your runtime after installation.
"""
