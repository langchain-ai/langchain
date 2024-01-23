from typing import Any, Dict, List, Mapping, Optional, cast

import pytest
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import validator

from langchain_core.documents.copy_transformer import CopyDocumentTransformer

TEMPERATURE = 0.0
MAX_TOKENS = 1000
FAKE_LLM = True
USE_CACHE = True




class FakeLLM(LLM):
    """Fake LLM wrapper for testing purposes."""

    queries: Optional[Mapping] = None
    sequential_responses: Optional[bool] = False
    response_index: int = 0

    @validator("queries", always=True)
    def check_queries_required(
        cls, queries: Optional[Mapping], values: Mapping[str, Any]
    ) -> Optional[Mapping]:
        if values.get("sequential_response") and not queries:
            raise ValueError(
                "queries is required when sequential_response is set to True"
            )
        return queries

    def get_num_tokens(self, text: str) -> int:
        """Return number of tokens."""
        return len(text.split())

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.sequential_responses:
            return self._get_next_response_in_sequence
        if self.queries is not None:
            return self.queries[prompt]
        if stop is None:
            return "foo"
        else:
            return "bar"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {}

    @property
    def _get_next_response_in_sequence(self) -> str:
        queries = cast(Mapping, self.queries)
        response = queries[list(queries.keys())[self.response_index]]
        self.response_index = self.response_index + 1
        return response

def init_llm(
    queries: Dict[int, str],
    max_token: int = MAX_TOKENS,
) -> BaseLLM:
    if FAKE_LLM:
        return FakeLLM(
            queries=queries,
            sequential_responses=True,
        )
    else:
        import langchain
        from dotenv import load_dotenv
        from langchain_community.cache import SQLiteCache

        load_dotenv()

        if USE_CACHE:
            langchain.llm_cache = SQLiteCache(
                database_path="/tmp/cache_qa_with_reference.db"
            )
        llm = langchain.OpenAI(
            temperature=TEMPERATURE,
            max_tokens=max_token,
            # cache=False,
        )
        return llm


# %% copy_transformer
def test_copy_transformer_transform_documents() -> None:
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    result = CopyDocumentTransformer().transform_documents([doc1, doc2])
    assert len(result) == 2
    assert id(result[0]) != id(doc1)
    assert id(result[1]) != id(doc2)
    assert result[0] == doc1
    assert result[1] == doc2


def test_copy_transformer_lazy_transform_documents() -> None:
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    result = list(
        CopyDocumentTransformer().lazy_transform_documents(iter([doc1, doc2]))
    )
    assert len(result) == 2
    assert id(result[0]) != id(doc1)
    assert id(result[1]) != id(doc2)
    assert result[0] == doc1
    assert result[1] == doc2


@pytest.mark.asyncio
async def test_copy_transformer_atransform_documents() -> None:
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    result = await CopyDocumentTransformer().atransform_documents([doc1, doc2])
    assert len(result) == 2
    assert id(result[0]) != id(doc1)
    assert id(result[1]) != id(doc2)
    assert result[0] == doc1
    assert result[1] == doc2


@pytest.mark.asyncio
async def test_copy_transformer_alazy_transform_documents() -> None:
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    result = [
        doc
        async for doc in CopyDocumentTransformer().alazy_transform_documents(
            iter([doc1, doc2])
        )
    ]
    assert len(result) == 2
    assert id(result[0]) != id(doc1)
    assert id(result[1]) != id(doc2)
    assert result[0] == doc1
    assert result[1] == doc2


