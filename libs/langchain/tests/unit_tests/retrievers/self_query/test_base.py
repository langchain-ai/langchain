from typing import Any, Dict, List, Tuple, Union

import pytest

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers import SelfQueryRetriever
from langchain.schema import Document
from tests.unit_tests.indexes.test_indexing import InMemoryVectorStore
from tests.unit_tests.llms.fake_llm import FakeLLM


class FakeTranslator(Visitor):
    allowed_comparators = (
        Comparator.EQ,
        Comparator.NE,
        Comparator.LT,
        Comparator.LTE,
        Comparator.GT,
        Comparator.GTE,
        Comparator.CONTAIN,
        Comparator.LIKE,
    )
    allowed_operators = (Operator.AND, Operator.OR, Operator.NOT)

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        self._validate_func(func)
        return f"${func.value}"

    def visit_operation(self, operation: Operation) -> Dict:
        args = [arg.accept(self) for arg in operation.arguments]
        return {self._format_func(operation.operator): args}

    def visit_comparison(self, comparison: Comparison) -> Dict:
        return {
            comparison.attribute: {
                self._format_func(comparison.comparator): comparison.value
            }
        }

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"filter": structured_query.filter.accept(self)}
        return structured_query.query, kwargs


class InMemoryVectorstoreWithSearch(InMemoryVectorStore):
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        res = self.store.get(query)
        if res is None:
            return []
        return [res]


@pytest.fixture()
def fake_llm() -> FakeLLM:
    return FakeLLM(
        queries={
            "1": """```json
{
    "query": "test",
    "filter": null
}
```""",
            "bar": "baz",
        },
        sequential_responses=True,
    )


@pytest.fixture()
def fake_vectorstore() -> InMemoryVectorstoreWithSearch:
    vectorstore = InMemoryVectorstoreWithSearch()
    vectorstore.add_documents(
        [
            Document(
                page_content="test",
                metadata={
                    "foo": "bar",
                },
            ),
        ],
        ids=["test"],
    )
    return vectorstore


@pytest.fixture()
def fake_self_query_retriever(
    fake_llm: FakeLLM, fake_vectorstore: InMemoryVectorstoreWithSearch
) -> SelfQueryRetriever:
    return SelfQueryRetriever.from_llm(
        llm=fake_llm,
        vectorstore=fake_vectorstore,
        document_contents="test",
        metadata_field_info=[
            AttributeInfo(
                name="foo",
                type="string",
                description="test",
            ),
        ],
        structured_query_translator=FakeTranslator(),
    )


def test__get_relevant_documents(fake_self_query_retriever: SelfQueryRetriever) -> None:
    relevant_documents = fake_self_query_retriever._get_relevant_documents(
        "foo",
        run_manager=CallbackManagerForRetrieverRun.get_noop_manager(),
    )
    assert len(relevant_documents) == 1
    assert relevant_documents[0].metadata["foo"] == "bar"


@pytest.mark.asyncio
async def test__aget_relevant_documents(
    fake_self_query_retriever: SelfQueryRetriever,
) -> None:
    relevant_documents = await fake_self_query_retriever._aget_relevant_documents(
        "foo",
        run_manager=AsyncCallbackManagerForRetrieverRun.get_noop_manager(),
    )
    assert len(relevant_documents) == 1
    assert relevant_documents[0].metadata["foo"] == "bar"
