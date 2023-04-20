import os
import tempfile
import threading
from typing import List

import certifi
import tiktoken
import tiktoken_ext
from pyfakefs.fake_filesystem_unittest import Patcher

from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma

"""
pytest --capture=no --timeout=3  --log-cli-level=DEBUG -vvv tests/integration_tests/vectorstores_new/test_chroma_stuck.py
"""  # noqa: E501

os.environ["TIKTOKEN_CACHE_DIR"] = os.path.join(
    tempfile.gettempdir(), ".tiktoken_cache"
)

if not os.path.exists(os.environ["TIKTOKEN_CACHE_DIR"]):
    os.makedirs(os.environ["TIKTOKEN_CACHE_DIR"])


def print_threads(msg: str = "") -> None:
    print("#############################################")
    print(msg)
    for t in threading.enumerate():
        print(t)

        print(f"Name: {t.name}")
        print(f"Ident: {t.ident}")
        print(f"Alive: {t.is_alive()}")
        print(f"Daemon: {t.daemon}")
    print("#############################################")


class TestManualChroma:
    """
    Tests the Chroma vector store's static methods to ensure they work correctly with a
    local static vector store that does not persist data to disk.
    """

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

        encoding = tiktoken.get_encoding("cl100k_base")
        print(encoding.encode("tiktoken is great!"))

        print_threads("before patcher:")

        with Patcher(use_cache=False) as patcher:
            patcher.fs.add_real_directory(certifi.where(), read_only=True)
            patcher.fs.add_real_directory(
                os.environ["TIKTOKEN_CACHE_DIR"], read_only=False
            )

            patcher.fs.add_real_directory(
                tiktoken_ext.__path__._path[0], read_only=True
            )

            docsearch = Chroma.from_texts(
                texts=texts,
                embedding=embedding,
            )

            docsearch.similarity_search("sharks", 1)

        print_threads("after patcher:")

        for t in threading.enumerate():
            if t.name == "MainThread":
                continue
            t.daemon = False
        print("end")
