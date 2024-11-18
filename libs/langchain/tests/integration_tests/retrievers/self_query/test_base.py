"""Integration testing of retrievers/self_query/base.py"""

import pytest

from langchain.retrievers.self_query import base as B


@pytest.mark.requires("langchain_weaviate")
def test_weaviate_init() -> None:
    """"""
    llm = None
    vectorstore = None
    metadata_field_info = [{"name": "foo", "type": "string", "description": "test"}]

    B.SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents="test",
        metadata_field_info=metadata_field_info,
        verbose=True,
    )
