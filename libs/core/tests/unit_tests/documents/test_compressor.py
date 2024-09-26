from collections.abc import Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document


class FakeDocumentCompressorA(BaseDocumentCompressor):
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks = None,
    ) -> Sequence[Document]:
        if len(documents) > 0:
            return documents[1:]
        return []


class FakeDocumentCompressorB(BaseDocumentCompressor):
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks = None,
    ) -> Sequence[Document]:
        return [Document(page_content=doc.page_content.upper()) for doc in documents]


def test_fake_compressor() -> None:
    assert (
        len(
            FakeDocumentCompressorA().compress_documents(
                [Document(page_content="Hello, World!")] * 2, "query"
            )
        )
        == 1
    )
    assert (
        FakeDocumentCompressorB()
        .compress_documents([Document(page_content="Hello, World!")], "query")[0]
        .page_content.isupper()
    )


def test_compressor_sequence() -> None:
    compressor_sequence = FakeDocumentCompressorA() | FakeDocumentCompressorB()
    compress_result = compressor_sequence.compress_documents(
        [Document(page_content="Hello, World!")] * 2, "query"
    )
    assert len(compress_result) == 1
    assert compress_result[0].page_content.isupper()
