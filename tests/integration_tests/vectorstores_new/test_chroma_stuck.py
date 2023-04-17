import os
import tempfile
from typing import Generator, List

import certifi
import pytest
import tiktoken
from pyfakefs.fake_filesystem_unittest import Patcher

from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma

"""
pytest --capture=no --timeout=30 --log-cli-level=DEBUG -vvv tests/integration_tests/vectorstores_new/test_chroma_stuck.py
"""  # noqa: E501


@pytest.fixture(scope="function")
def texts() -> Generator[List[str], None, None]:
    # Load the texts from a file located in the fixtures directory
    documents = TextLoader(
        os.path.join(os.path.dirname(__file__), "fixtures", "sharks.txt")
    ).load()

    yield [doc.page_content for doc in documents]


# Define a fixture that returns an instance of the OpenAIEmbeddings class
@pytest.fixture(scope="function")
def embedding() -> OpenAIEmbeddings:
    # Double check that the API key is set
    assert os.getenv("OPENAI_API_KEY") is not None
    return OpenAIEmbeddings()


# Define a fixture that returns a query string to use for testing
@pytest.fixture(scope="module")
def query() -> str:
    return "sharks"


class TestManualChroma:
    """
    Tests the Chroma vector store's static methods to ensure they work correctly with a
    local static vector store that does not persist data to disk.
    """

    # @pytest.fixture(autouse=True)
    # def _prepare_tiktoken(self) -> None:
    #     print("")
    #     print("prepare_tiktoken start")
    #     encoding = tiktoken.get_encoding("cl100k_base")
    #     print(encoding.encode("tiktoken is great!"))
    #
    #     yield
    #
    #     print("prepare_tiktoken end")

    @classmethod
    def teardown_class(cls) -> None:
        print("teardown_class")

    def test_from_texts(
            self, texts: List[str], embedding: Embeddings, query: str
    ) -> None:
        """
        Test creating a VectorStore from a list of texts.
        """
        print("test_from_texts")
        additional_skip_names = [tiktoken,tiktoken.registry,tiktoken.core,tiktoken.model]

        with Patcher(additional_skip_names=additional_skip_names) as patcher:
            patcher.fs.add_real_directory(certifi.where(), read_only=True)
            # Because tiktoken saved downloaded models in temp dir
            patcher.fs.add_mount_point(tempfile.gettempdir(), can_exist=True)
            print("1 start")

            self.docsearch = Chroma.from_texts(
                texts=texts,
                embedding=embedding,
            )

            self.docsearch._client.reset()
            self.docsearch = None
            print("1 end")

        print("end")
