"""Filter that uses an LLM to drop documents that aren't relevant to the query."""

from collections.abc import Sequence
from typing import Any, Callable, Optional

from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from pydantic import ConfigDict

from langchain.chains import LLMChain
from langchain.output_parsers.boolean import BooleanOutputParser
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.retrievers.document_compressors.chain_filter_prompt import (
    prompt_template,
)


def _get_default_chain_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=prompt_template,
        input_variables=["question", "context"],
        output_parser=BooleanOutputParser(),
    )


def default_get_input(query: str, doc: Document) -> dict[str, Any]:
    """Return the compression chain input."""
    return {"question": query, "context": doc.page_content}


class LLMChainFilter(BaseDocumentCompressor):
    """Filter that drops documents that aren't relevant to the query."""

    llm_chain: Runnable
    """LLM wrapper to use for filtering documents. 
    The chain prompt is expected to have a BooleanOutputParser."""

    get_input: Callable[[str, Document], dict] = default_get_input
    """Callable for constructing the chain input from the query and a Document."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Filter down documents based on their relevance to the query."""
        filtered_docs = []

        config = RunnableConfig(callbacks=callbacks)
        outputs = zip(
            self.llm_chain.batch(
                [self.get_input(query, doc) for doc in documents], config=config
            ),
            documents,
        )

        for output_, doc in outputs:
            include_doc = None
            if isinstance(self.llm_chain, LLMChain):
                output = output_[self.llm_chain.output_key]
                if self.llm_chain.prompt.output_parser is not None:
                    include_doc = self.llm_chain.prompt.output_parser.parse(output)
            else:
                if isinstance(output_, bool):
                    include_doc = output_
            if include_doc:
                filtered_docs.append(doc)

        return filtered_docs

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Filter down documents based on their relevance to the query."""
        filtered_docs = []

        config = RunnableConfig(callbacks=callbacks)
        outputs = zip(
            await self.llm_chain.abatch(
                [self.get_input(query, doc) for doc in documents], config=config
            ),
            documents,
        )
        for output_, doc in outputs:
            include_doc = None
            if isinstance(self.llm_chain, LLMChain):
                output = output_[self.llm_chain.output_key]
                if self.llm_chain.prompt.output_parser is not None:
                    include_doc = self.llm_chain.prompt.output_parser.parse(output)
            else:
                if isinstance(output_, bool):
                    include_doc = output_
            if include_doc:
                filtered_docs.append(doc)

        return filtered_docs

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[BasePromptTemplate] = None,
        **kwargs: Any,
    ) -> "LLMChainFilter":
        """Create a LLMChainFilter from a language model.

        Args:
            llm: The language model to use for filtering.
            prompt: The prompt to use for the filter.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            A LLMChainFilter that uses the given language model.
        """
        _prompt = prompt if prompt is not None else _get_default_chain_prompt()
        if _prompt.output_parser is not None:
            parser = _prompt.output_parser
        else:
            parser = StrOutputParser()
        llm_chain = _prompt | llm | parser
        return cls(llm_chain=llm_chain, **kwargs)
