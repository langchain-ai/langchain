import copy
from typing import Any, AsyncIterator, Callable, Dict, Iterator, Optional

from langchain.chains import LLMChain
from langchain.document_transformers.runnable_document_transformer import (
    RunnableGeneratorDocumentTransformer,
)
from langchain.output_parsers import NumberedListOutputParser

# Note: Import directly from langchain_core is not stable and generate some errors
# from langchain_core.prompts import PromptTemplate
# from langchain_core.documents import Document
# from langchain_core.language_models import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema.language_model import BaseLanguageModel


def _default_get_input(doc: Document) -> Dict[str, Any]:
    """Return the context chain input."""
    return {
        "context": doc.page_content,
    }


_default_template = """
Summarize a text input in the same language. 
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


class SummarizeTransformer(RunnableGeneratorDocumentTransformer):
    """Generate questions for each Documents."""

    llm_chain: LLMChain
    get_input: Callable[[Document], dict] = _default_get_input

    """LLM wrapper to use for compressing documents."""

    """Callable for constructing the chain input from the query and a Document."""

    def lazy_transform_documents(
        self, documents: Iterator[Document], **kwargs: Any
    ) -> Iterator[Document]:
        _callbacks = kwargs.get("callbacks", None)
        for doc in documents:
            _input = self.get_input(doc)
            output = self.llm_chain.predict(callbacks=_callbacks, **_input)
            if not output:
                continue
            metadata = copy.deepcopy(doc.metadata)
            metadata["transformer"] = self.__class__.__name__
            yield Document(
                page_content="SUMMARY:\n" + str(output).strip(), metadata=metadata
            )

    #
    # def transform_documents(
    #         self, documents: Sequence[Document], **kwargs: Any
    # ) -> Sequence[Document]:
    #     return list(self.lazy_transform_documents(
    #     documents=iter(documents), **kwargs))
    #
    async def _alazy_transform_documents(  # type:ignore
        self, documents: AsyncIterator[Document], **kwargs: Any
    ) -> AsyncIterator[Document]:
        _callbacks = kwargs.get("callbacks", None)
        async for doc in documents:
            _input = self.get_input(doc)
            output = await self.llm_chain.apredict(callbacks=_callbacks, **_input)
            if not output:
                continue
            metadata = copy.deepcopy(doc.metadata)
            metadata["transformer"] = self.__class__.__name__
            yield Document(
                page_content="SUMMARY:\n" + str(output).strip(), metadata=metadata
            )

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[PromptTemplate] = None,
        get_input: Optional[Callable[[Document], dict]] = None,
        llm_chain_kwargs: Optional[dict] = None,
    ) -> "SummarizeTransformer":
        """Initialize from LLM."""
        _prompt = prompt if prompt is not None else _get_default_chain_prompt()
        _get_input = get_input if get_input is not None else _default_get_input
        llm_chain = LLMChain(
            llm=llm,
            prompt=_prompt,
            **(llm_chain_kwargs or {}),
        )
        return cls(llm_chain=llm_chain, get_input=_get_input)
