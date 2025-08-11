from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.callbacks.mlflow_callback import (
        MlflowCallbackHandler,
        MlflowLogger,
        analyze_text,
        construct_html_from_prompt_and_generation,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "analyze_text": "langchain_community.callbacks.mlflow_callback",
    "construct_html_from_prompt_and_generation": (
        "langchain_community.callbacks.mlflow_callback"
    ),
    "MlflowLogger": "langchain_community.callbacks.mlflow_callback",
    "MlflowCallbackHandler": "langchain_community.callbacks.mlflow_callback",
}

_import_attribute = create_importer(__file__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "MlflowCallbackHandler",
    "MlflowLogger",
    "analyze_text",
    "construct_html_from_prompt_and_generation",
]
