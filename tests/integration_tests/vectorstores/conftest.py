import os
from typing import Generator, List

import pytest

from langchain.document_loaders import TextLoader
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter


@pytest.fixture(scope="module")
def documents() -> Generator[List[Document], None, None]:
    """Return a generator that yields a list of documents."""
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    documents = TextLoader(
        os.path.join(os.path.dirname(__file__), "fixtures", "sharks.txt")
    ).load()
    yield text_splitter.split_documents(documents)
