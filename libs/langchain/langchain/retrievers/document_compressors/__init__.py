import importlib
from typing import Any

from langchain.retrievers.document_compressors.base import DocumentCompressorPipeline
from langchain.retrievers.document_compressors.chain_extract import (
    LLMChainExtractor,
)
from langchain.retrievers.document_compressors.chain_filter import (
    LLMChainFilter,
)
from langchain.retrievers.document_compressors.cohere_rerank import CohereRerank
from langchain.retrievers.document_compressors.cross_encoder_rerank import (
    CrossEncoderReranker,
)
from langchain.retrievers.document_compressors.embeddings_filter import (
    EmbeddingsFilter,
)
from langchain.retrievers.document_compressors.listwise_rerank import (
    LLMListwiseRerank,
)

_module_lookup = {
    "FlashrankRerank": "langchain_community.document_compressors.flashrank_rerank",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    msg = f"module {__name__} has no attribute {name}"
    raise AttributeError(msg)


__all__ = [
    "CohereRerank",
    "CrossEncoderReranker",
    "DocumentCompressorPipeline",
    "EmbeddingsFilter",
    "FlashrankRerank",
    "LLMChainExtractor",
    "LLMChainFilter",
    "LLMListwiseRerank",
]
