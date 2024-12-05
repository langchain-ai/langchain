"""Integration testing of retrievers/self_query/base.py

########################################
# Here is a Weaviate v4 example:
# Step 1: Launch the Weaviate v4 container

cd tests/integration_tests/vectorstores/docker-compose
docker compose -f weaviate.yml up

# Step 2: Run the test
pytest -sv tests/integration_tests/retrievers/self_query/test_base.py

########################################

"""

import os

import pytest
import weaviate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.classes.config import DataType, Property

from langchain.retrievers.self_query import base as B


@pytest.fixture(scope="module")
def embedding_openai() -> OpenAIEmbeddings:
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set")
    return OpenAIEmbeddings()


@pytest.fixture(scope="module")
def chat_model_openai() -> ChatOpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set")

    return ChatOpenAI(model="gpt-4o", temperature=0)


class TestWeaviate:
    @classmethod
    def setup_class(cls) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")

    @pytest.fixture(scope="class", autouse=True)
    def weaviate_host(self) -> str:  # type: ignore[return]
        # localhost or weaviate
        host = "localhost"
        yield host

    @pytest.fixture(scope="class", autouse=True)
    def weaviate_port(self) -> str:  # type: ignore[return]
        port = "8080"
        yield port

    @pytest.fixture(scope="class", autouse=True)
    def weaviate_client(
        self, weaviate_host: str, weaviate_port: str
    ) -> weaviate.WeaviateClient:
        weaviate_client = weaviate.connect_to_local(
            host=weaviate_host, port=weaviate_port
        )
        # Create the collection if not exists
        # The collection name is assumed to be "test" throughout the test module
        if not weaviate_client.collections.exists("test"):
            weaviate_client.collections.create(
                name="test", properties=[Property(name="name", data_type=DataType.TEXT)]
            )
        yield weaviate_client

    def test_self_query_retriever(
        self,
        weaviate_client: str,
        embedding_openai: OpenAIEmbeddings,
        chat_model_openai: ChatOpenAI,
    ) -> None:
        """Test end to end construction and search with metadata."""

        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        vectorstore = WeaviateVectorStore.from_texts(
            texts, embedding_openai, metadatas=metadatas, client=weaviate_client
        )

        metadata_field_info = [
            {
                "name": "product_name",
                "type": "string",
                "description": "Short product name",
            }
        ]
        retriever = B.SelfQueryRetriever.from_llm(
            llm=chat_model_openai,
            vectorstore=vectorstore,
            document_contents="Product names and associated attributes",
            metadata_field_info=metadata_field_info,
            verbose=True,
        )
        input_question = "A product name similar to 'foo'"

        vectorstore.similarity_search(input_question, k=1)
        retriever.invoke(input_question)
        retriever.get_relevant_documents(input_question)
