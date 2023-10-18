import asyncio
from functools import partial
from typing import Callable, Sequence, Optional, Dict, Any, Generator

from langchain.chains import LLMChain
from langchain.output_parsers import NumberedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document, BaseDocumentTransformer
from langchain.schema.language_model import BaseLanguageModel


def _default_get_input(doc: Document) -> Dict[str, Any]:
    """Return the context chain input."""
    return {
        "context": doc.page_content,
    }


_default_template = """
Sumarize a text input in the same language. 
Context:
```
{context}
```
"""
_default_format_instruction = NumberedListOutputParser()


def _get_default_chain_prompt() -> PromptTemplate:
    return PromptTemplate.from_template(
        template=_default_template,
    )


class SummarizeTransformer(BaseDocumentTransformer):
    """Generate questions for each Documents."""

    def __init__(
            self,
            llm_chain: LLMChain,
            get_input: Callable[[Document], dict] = _default_get_input,
    ):
        self.llm_chain = llm_chain
        self.get_input = get_input

    """LLM wrapper to use for compressing documents."""

    """Callable for constructing the chain input from the query and a Document."""

    def lazy_transform_documents(
            self,
            documents: Sequence[Document],
            **kwargs: Any,
    ) -> Generator[Document, None, None]:
        """Compress page content of raw documents."""
        _callbacks = kwargs.get("callbacks", None)
        for doc in documents:
            _input = self.get_input(doc)
            output = self.llm_chain.predict_and_parse(
                callbacks=_callbacks,
                **_input)
            if not output:
                continue
            yield Document(page_content=str(output), metadata=doc.metadata)

    def transform_documents(
            self,
            documents: Sequence[Document],
            **kwargs: Any
    ) -> Sequence[Document]:
        return list(self.lazy_transform_documents(
            documents=documents,
            **kwargs
        ))

    async def lazy_atransform_documents(
            self, documents: Sequence[Document], **kwargs: Any
    ) -> Generator[Document, None, None]:

        """Summarize the page content of raw documents asynchronously."""
        _callbacks = kwargs.get("callbacks", None)
        outputs = await asyncio.gather(
            *[
                self.llm_chain.apredict_and_parse(
                    **self.get_input(documents),
                    callbacks=_callbacks
                )
                for doc in documents
            ]
        )
        for i, doc in enumerate(documents):
            if not outputs[i]:
                continue
            yield Document(page_content=outputs[i],
                           metadata=doc.metadata)

    async def atransform_documents(
            self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Asynchronously transform a list of documents.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            A list of transformed Documents.
        """
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self.transform_documents, **kwargs), documents
        )

    @classmethod
    def from_llm(
            cls,
            llm: BaseLanguageModel,
            prompt: Optional[PromptTemplate] = None,
            get_input: Optional[Callable[[Document], dict]] = None,
            llm_chain_kwargs: Optional[dict] = None,
    ) -> 'SummarizeTransformer':
        """Initialize from LLM."""
        _prompt = prompt if prompt is not None else _get_default_chain_prompt()
        _get_input = get_input if get_input is not None else _default_get_input
        llm_chain = LLMChain(llm=llm, prompt=_prompt, **(llm_chain_kwargs or {}))
        return cls(llm_chain=llm_chain,
                   get_input=_get_input)
