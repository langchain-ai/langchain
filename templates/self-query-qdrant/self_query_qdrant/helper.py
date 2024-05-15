from string import Formatter
from typing import List

from langchain_core.documents import Document

document_template = """
PASSAGE: {page_content}
METADATA: {metadata}
"""


def combine_documents(documents: List[Document]) -> str:
    """
    Combine a list of documents into a single string that might be passed further down
    to a language model.
    :param documents: list of documents to combine
    :return:
    """
    formatter = Formatter()
    return "\n\n".join(
        formatter.format(
            document_template,
            page_content=document.page_content,
            metadata=document.metadata,
        )
        for document in documents
    )
