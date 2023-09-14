"""Filter that uses an LLM to drop documents that aren't relevant to the query."""
from typing import Any, Callable, Dict, Optional, Sequence

from langchain import LLMChain, PromptTemplate
from langchain.callbacks.manager import Callbacks
from langchain.output_parsers.boolean import BooleanOutputParser
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.retrievers.document_compressors.chain_filter_prompt import (
    prompt_template,
)
from langchain.schema import BasePromptTemplate, Document
from langchain.schema.language_model import BaseLanguageModel


def _get_default_chain_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=prompt_template,
        input_variables=["question", "context"],
        output_parser=BooleanOutputParser(),
    )


def default_get_input(query: str, doc: Document) -> Dict[str, Any]:
    """Return the compression chain input."""
    return {"question": query, "context": doc.page_content}


class LLMChainFilter(BaseDocumentCompressor):
    """Filter that drops documents that aren't relevant to the query."""

    llm_chain: LLMChain
    """LLM wrapper to use for filtering documents. 
    The chain prompt is expected to have a BooleanOutputParser."""

    get_input: Callable[[str, Document], dict] = default_get_input
    """Callable for constructing the chain input from the query and a Document."""

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Filter down documents based on their relevance to the query."""
        filtered_docs = []
        for doc in documents:
            _input = self.get_input(query, doc)
            include_doc = self.llm_chain.predict_and_parse(
                **_input, callbacks=callbacks
            )
            if include_doc:
                filtered_docs.append(doc)
        return filtered_docs

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Filter down documents."""
        raise NotImplementedError()

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[BasePromptTemplate] = None,
        **kwargs: Any
    ) -> "LLMChainFilter":
        """Create a LLMChainFilter from a language model.

        Args:
            llm: The language model to use for filtering.
            prompt: The prompt to use for the filter.
            **kwargs: Additional arguments to pass to the constructor.

        Returns:
            A LLMChainFilter that uses the given language model.
        """
        _prompt = prompt if prompt is not None else _get_default_chain_prompt()
        llm_chain = LLMChain(llm=llm, prompt=_prompt)
        return cls(llm_chain=llm_chain, **kwargs)
