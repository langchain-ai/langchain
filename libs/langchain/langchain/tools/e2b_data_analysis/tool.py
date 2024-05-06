from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import E2BDataAnalysisTool
    from langchain_community.tools.e2b_data_analysis.tool import (
        E2BDataAnalysisToolArguments,
        UploadedFile,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "UploadedFile": "langchain_community.tools.e2b_data_analysis.tool",
    "E2BDataAnalysisToolArguments": "langchain_community.tools.e2b_data_analysis.tool",
    "E2BDataAnalysisTool": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "UploadedFile",
    "E2BDataAnalysisToolArguments",
    "E2BDataAnalysisTool",
]
