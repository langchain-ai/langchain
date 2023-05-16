from typing import List

from langchain.schema import BaseRetriever, Document
from langchain.utilities.bibtex import BibtexparserWrapper


class BibtexRetriever(BaseRetriever, BibtexparserWrapper):
    """
    It is effectively a wrapper for BibtexparserWrapper.
    It wraps load() to get_relevant_documents().
    It uses all BibtexparserWrapper arguments without any change.
    """

    def get_relevant_documents(self, file_path: str) -> List[Document]:
        return self.load(file_path=file_path)

    async def aget_relevant_documents(self, file_path: str) -> List[Document]:
        raise NotImplementedError
