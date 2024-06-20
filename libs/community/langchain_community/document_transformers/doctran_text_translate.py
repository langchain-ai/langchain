from __future__ import annotations

import asyncio
from typing import Any, Optional, Sequence

from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.runnables.config import run_in_executor
from langchain_core.utils import get_from_env


class DoctranTextTranslator(BaseDocumentTransformer):
    """Translate text documents using doctran.

    Arguments:
        openai_api_key: OpenAI API key. Can also be specified via environment variable
        ``OPENAI_API_KEY``.
        language: The language to translate *to*.

    Example:
        .. code-block:: python

        from langchain_community.document_transformers import DoctranTextTranslator

        # Pass in openai_api_key or set env var OPENAI_API_KEY
        qa_translator = DoctranTextTranslator(language="spanish")
        translated_document = await qa_translator.atransform_documents(documents)
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        language: str = "english",
        openai_api_model: Optional[str] = None,
    ) -> None:
        self.openai_api_key = openai_api_key or get_from_env(
            "openai_api_key", "OPENAI_API_KEY"
        )
        self.openai_api_model = openai_api_model or get_from_env(
            "openai_api_model", "OPENAI_API_MODEL"
        )
        self.language = language

    async def _aparse_document(
        self, doctran: Any, index: int, doc: Document
    ) -> tuple[int, Any]:
        parsed_doc = await run_in_executor(
            None, doctran.parse, content=doc.page_content, metadata=doc.metadata
        )
        return index, parsed_doc

    async def _atranslate_document(
        self, index: int, doc: Any, language: str
    ) -> tuple[int, Any]:
        translated_doc = await run_in_executor(
            None, lambda: doc.translate(language=language).execute()
        )
        return index, translated_doc

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Translates text documents using doctran."""
        try:
            from doctran import Doctran

            doctran = Doctran(
                openai_api_key=self.openai_api_key, openai_model=self.openai_api_model
            )
        except ImportError:
            raise ImportError(
                "Install doctran to use this parser. (pip install doctran)"
            )

        parse_tasks = [
            self._aparse_document(doctran, i, doc) for i, doc in enumerate(documents)
        ]
        parsed_results = await asyncio.gather(*parse_tasks)

        parsed_results.sort(key=lambda x: x[0])
        doctran_docs = [doc for _, doc in parsed_results]

        translate_tasks = [
            self._atranslate_document(i, doc, self.language)
            for i, doc in enumerate(doctran_docs)
        ]
        translated_results = await asyncio.gather(*translate_tasks)

        translated_results.sort(key=lambda x: x[0])
        translated_docs = [doc for _, doc in translated_results]

        return [
            Document(page_content=doc.transformed_content, metadata=doc.metadata)
            for doc in translated_docs
        ]

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Translates text documents using doctran."""
        try:
            from doctran import Doctran

            doctran = Doctran(
                openai_api_key=self.openai_api_key, openai_model=self.openai_api_model
            )
        except ImportError:
            raise ImportError(
                "Install doctran to use this parser. (pip install doctran)"
            )
        doctran_docs = [
            doctran.parse(content=doc.page_content, metadata=doc.metadata)
            for doc in documents
        ]
        for i, doc in enumerate(doctran_docs):
            doctran_docs[i] = doc.translate(language=self.language).execute()
        return [
            Document(page_content=doc.transformed_content, metadata=doc.metadata)
            for doc in doctran_docs
        ]
