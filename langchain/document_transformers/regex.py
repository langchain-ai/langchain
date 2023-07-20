import re
from typing import Any, Sequence

from langchain.schema import BaseDocumentTransformer, Document


class RegexTransformer(BaseDocumentTransformer):
    """Replace occurrences of a particular search pattern with a replacement string

    Example:
        .. code-block:: python

            from langchain.document_transformers import RegexTransformer
            re=RegexTransformer(regex="\n",replacement_str="")
            docs_transform=re.transform_documents(docs)
    """

    def __init__(self, regex: str, replacement_str: str = ""):
        self.regex = regex
        self.replacement_str = replacement_str

    def transform_documents(
        self,
        documents: Sequence[Document],
        **kwargs: Any,
    ) -> Sequence[Document]:
        for d in documents:
            d.page_content = re.sub(self.regex, self.replacement_str, d.page_content)
        return documents

    async def atransform_documents(
        self,
        documents: Sequence[Document],
        **kwargs: Any,
    ) -> Sequence[Document]:
        raise NotImplementedError
