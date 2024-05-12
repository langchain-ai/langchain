import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_community.document_compressors.jina_rerank import (
        JinaRerank,  # noqa: F401
    )
    from langchain_community.document_compressors.llmlingua_filter import (
        LLMLinguaCompressor,
    )
    from langchain_community.document_compressors.openvino_rerank import (
        OpenVINOReranker,
    )

__all__ = ["LLMLinguaCompressor", "OpenVINOReranker"]

_module_lookup = {
    "LLMLinguaCompressor": "langchain_community.document_compressors.llmlingua_filter",
    "OpenVINOReranker": "langchain_community.document_compressors.openvino_rerank",
    "JinaRerank": "langchain_community.document_compressors.jina_rerank",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_module_lookup.keys())
