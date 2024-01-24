import os
import tempfile
from typing import Any, Sequence

import pytest
from langchain_core.documents import BaseDocumentTransformer, Document

from langchain_community.document_loaders import TextLoader

try:
    import langchain

    is_langchain = langchain is not None
except ImportError:
    is_langchain = False


def test_load_and_transform() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test.txt
        filename = os.path.join(temp_dir, "test.txt")
        with open(filename, "w") as test_txt:
            test_txt.write("This is a test.txt file.")
        loader = TextLoader(os.path.join(temp_dir, "test.txt"))

        class UpperTransformer(BaseDocumentTransformer):
            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                return [
                    Document(
                        page_content=doc.page_content.upper(), metadata=doc.metadata
                    )
                    for doc in documents
                ]

            async def atransform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                return [
                    Document(page_content=doc.page_content.upper()) for doc in documents
                ]

        assert loader.load_and_transform(UpperTransformer()) == [
            Document(
                page_content="THIS IS A TEST.TXT FILE.", metadata={"source": filename}
            )
        ]


@pytest.mark.skipif(not is_langchain, reason="Valid only with 'langchain'")
def test_load_and_split() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test.txt
        filename = os.path.join(temp_dir, "test.txt")
        with open(filename, "w") as test_txt:
            test_txt.write("This is a test.txt file.")
        loader = TextLoader(os.path.join(temp_dir, "test.txt"))

        from langchain.text_splitter import CharacterTextSplitter

        assert loader.load_and_split(
            CharacterTextSplitter(chunk_size=20, chunk_overlap=0, separator=" ")
        ) == [
            Document(page_content="This is a test.txt", metadata={"source": filename}),
            Document(page_content="file.", metadata={"source": filename}),
        ]
