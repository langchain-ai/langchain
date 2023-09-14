from __future__ import annotations

from typing import Any, Sequence

from langchain.chains import ArcGISRowSummaryChain
from langchain.schema import BaseDocumentTransformer, Document


class ArcGISRowSummaryTransformer(BaseDocumentTransformer):
    """
    A class to produce bullet-point summaries for
    geospatial data rows using a Large Language Model.

    Attributes:
        chain (LLMChain): A chain of language models.
    """

    def __init__(
        self,
        chain: ArcGISRowSummaryChain,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.chain = chain

    @staticmethod
    def desc_from_doc(doc: Document) -> str:
        """
        Extracts the description from a document.

        Args:
            doc (Document): The document to extract the description from.

        Returns:
            str: The description.
        """
        item_desc = doc.metadata["item_description"]
        layer_desc = doc.metadata["layer_description"]
        return "\n".join((item_desc, layer_desc))

    @classmethod
    def docs_to_inputs(
        cls,
        docs: list[Document],
    ) -> list[dict[str, str]]:
        """
        Converts documents to input format for summarization.

        Args:
            docs (list[Document]): List of documents representing geospatial data rows.

        Returns:
            list[dict]: A list of dictionaries, each representing
            an input for the summarizer.
        """
        return [
            {
                "name": doc.metadata["name"],
                "desc": cls.desc_from_doc(doc),
                "json_str": doc.page_content,
            }
            for doc in docs
        ]

    def summarize(self, docs: list[Document]) -> list[str]:
        """
        Produces bullet-point summaries for given geospatial data rows.

        Args:
            docs (list[Document]): List of documents representing geospatial data rows.

        Returns:
            list[str]: A list of bullet-point summaries.
        """
        inputs = self.docs_to_inputs(docs)
        return [result["text"] for result in self.chain.apply(inputs)]

    async def asummarize(self, docs: list[Document]) -> list[str]:
        """
        Asynchronously produces bullet-point summaries for given geospatial data rows.

        Args:
            docs (list[Document]): List of documents representing geospatial data rows.

        Returns:
            list[str]: A list of bullet-point summaries.
        """
        inputs = self.docs_to_inputs(docs)
        return [result["text"] for result in await self.chain.aapply(inputs)]

    @classmethod
    def return_docs(
        cls, docs: Sequence[Document], summaries: Sequence[str]
    ) -> Sequence[Document]:
        """
        Returns documents with summaries.

        Args:
            docs (list[Document]): List of documents representing geospatial data rows.
            summaries (list[str]): List of summaries.

        Returns:
            list[Document]: List of documents with summaries.
        """
        for doc, summary in zip(docs, summaries):
            # convert attributes back to dict and store in metadata
            doc.metadata["attributes"] = json.loads(doc.page_content)
            # replace page_content with the generated summary
            doc.page_content = summary
        return docs

    def transform_documents(
        self,
        documents: Sequence[Document],
        **kwargs: Any,
    ) -> Sequence[Document]:
        summaries = self.summarize(list(documents))
        return self.return_docs(documents, summaries)

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        summaries = await self.asummarize(list(documents))
        return self.return_docs(documents, summaries)
