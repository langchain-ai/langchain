import os
from typing import Generator, List

import pytest

from langchain.document_loaders import TextLoader
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter


# Define a fixture that yields a generator object returning a list of documents
@pytest.fixture(scope="module")
def documents() -> Generator[List[Document], None, None]:
    """Return a generator that yields a list of documents."""

    # Create a CharacterTextSplitter object for splitting the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    # Load the documents from a file located in the fixtures directory
    documents = TextLoader(
        os.path.join(os.path.dirname(__file__), "fixtures", "sharks.txt")
    ).load()

    # Yield the documents split into chunks
    yield text_splitter.split_documents(documents)
