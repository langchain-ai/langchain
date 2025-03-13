from __future__ import annotations

from copy import deepcopy
from enum import Enum
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from packaging.version import Version
from pydantic import ConfigDict, Field, PrivateAttr, model_validator

if TYPE_CHECKING:
    from rank_llm.data import Candidate, Query, Request
else:
    # Avoid pydantic annotation issues when actually instantiating
    # while keeping this import optional
    try:
        from rank_llm.data import Candidate, Query, Request
    except ImportError:
        pass


class RankLLMRerank(BaseDocumentCompressor):
    """Document compressor using Flashrank interface."""

    client: Any = None
    """RankLLM client to use for compressing documents"""
    top_n: int = Field(default=3)
    """Top N documents to return."""
    model: str = Field(default="zephyr")
    """Name of model to use for reranking."""
    step_size: int = Field(default=10)
    """Step size for moving sliding window."""
    gpt_model: str = Field(default="gpt-3.5-turbo")
    """OpenAI model name."""
    _retriever: Any = PrivateAttr()

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate python package exists in environment."""

        if not values.get("client"):
            client_name = values.get("model", "zephyr")

            is_pre_rank_llm_revamp = Version(version=version("rank_llm")) <= Version(
                "0.12.8"
            )

            try:
                model_enum = ModelType(client_name.lower())
            except ValueError:
                raise ValueError(
                    "Unsupported model type. Please use 'vicuna', 'zephyr', or 'gpt'."
                )

            try:
                if model_enum == ModelType.VICUNA:
                    if is_pre_rank_llm_revamp:
                        from rank_llm.rerank.vicuna_reranker import VicunaReranker
                    else:
                        from rank_llm.rerank.listwise.vicuna_reranker import (
                            VicunaReranker,
                        )

                    values["client"] = VicunaReranker()
                elif model_enum == ModelType.ZEPHYR:
                    if is_pre_rank_llm_revamp:
                        from rank_llm.rerank.zephyr_reranker import ZephyrReranker
                    else:
                        from rank_llm.rerank.listwise.zephyr_reranker import (
                            ZephyrReranker,
                        )

                    values["client"] = ZephyrReranker()
                elif model_enum == ModelType.GPT:
                    if is_pre_rank_llm_revamp:
                        from rank_llm.rerank.rank_gpt import SafeOpenai
                    else:
                        from rank_llm.rerank.listwise.rank_gpt import SafeOpenai

                    from rank_llm.rerank.reranker import Reranker

                    openai_api_key = get_from_dict_or_env(
                        values, "open_api_key", "OPENAI_API_KEY"
                    )

                    agent = SafeOpenai(
                        model=values["gpt_model"],
                        context_size=4096,
                        keys=openai_api_key,
                    )
                    values["client"] = Reranker(agent)

            except ImportError:
                raise ImportError(
                    "Could not import rank_llm python package. "
                    "Please install it with `pip install rank_llm`."
                )

        return values

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        request = Request(
            query=Query(text=query, qid=1),
            candidates=[
                Candidate(doc={"text": doc.page_content}, docid=index, score=1)
                for index, doc in enumerate(documents)
            ],
        )

        rerank_results = self.client.rerank(
            request,
            rank_end=len(documents),
            window_size=min(20, len(documents)),
            step=10,
        )

        final_results = []
        if hasattr(rerank_results, "candidates"):
            # Old API format
            for res in rerank_results.candidates:
                doc = documents[int(res.docid)]
                doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
                final_results.append(doc_copy)
        else:
            for res in rerank_results:
                doc = documents[int(res.docid)]
                doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
                final_results.append(doc_copy)

        return final_results[: self.top_n]


class ModelType(Enum):
    VICUNA = "vicuna"
    ZEPHYR = "zephyr"
    GPT = "gpt"
