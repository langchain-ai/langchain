"""Base interface for chains combining documents."""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core._api import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.documents import Document
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_core.utils.pydantic import create_model
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from pydantic import BaseModel, Field
from typing_extensions import override

from langchain_classic.chains.base import Chain

DEFAULT_DOCUMENT_SEPARATOR = "\n\n"
DOCUMENTS_KEY = "context"
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template("{page_content}")


def _validate_prompt(prompt: BasePromptTemplate, document_variable_name: str) -> None:
    if document_variable_name not in prompt.input_variables:
        msg = (
            f"Prompt must accept {document_variable_name} as an input variable. "
            f"Received prompt with input variables: {prompt.input_variables}"
        )
        raise ValueError(msg)


class BaseCombineDocumentsChain(Chain, ABC):
    """Base interface for chains combining documents.

    Subclasses of this chain deal with combining documents in a variety of
    ways. This base class exists to add some uniformity in the interface these types
    of chains should expose. Namely, they expect an input key related to the documents
    to use (default `input_documents`), and then also expose a method to calculate
    the length of a prompt from documents (useful for outside callers to use to
    determine whether it's safe to pass a list of documents into this chain or whether
    that will be longer than the context length).
    """

    input_key: str = "input_documents"
    output_key: str = "output_text"

    @override
    def get_input_schema(
        self,
        config: RunnableConfig | None = None,
    ) -> type[BaseModel]:
        return create_model(
            "CombineDocumentsInput",
            **{self.input_key: (list[Document], None)},
        )

    @override
    def get_output_schema(
        self,
        config: RunnableConfig | None = None,
    ) -> type[BaseModel]:
        return create_model(
            "CombineDocumentsOutput",
            **{self.output_key: (str, None)},
        )

    @property
    def input_keys(self) -> list[str]:
        """Expect input key."""
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        """Return output key."""
        return [self.output_key]

    def prompt_length(self, docs: list[Document], **kwargs: Any) -> int | None:  # noqa: ARG002
        """Return the prompt length given the documents passed in.

        This can be used by a caller to determine whether passing in a list
        of documents would exceed a certain prompt length. This useful when
        trying to ensure that the size of a prompt remains below a certain
        context limit.

        Args:
            docs: a list of documents to use to calculate the total prompt length.
            **kwargs: additional parameters that may be needed to calculate the
                prompt length.

        Returns:
            Returns None if the method does not depend on the prompt length,
            otherwise the length of the prompt in tokens.
        """
        return None

    @abstractmethod
    def combine_docs(self, docs: list[Document], **kwargs: Any) -> tuple[str, dict]:
        """Combine documents into a single string.

        Args:
            docs: List[Document], the documents to combine
            **kwargs: Other parameters to use in combining documents, often
                other inputs to the prompt.

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        """

    @abstractmethod
    async def acombine_docs(
        self,
        docs: list[Document],
        **kwargs: Any,
    ) -> tuple[str, dict]:
        """Combine documents into a single string.

        Args:
            docs: List[Document], the documents to combine
            **kwargs: Other parameters to use in combining documents, often
                other inputs to the prompt.

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        """

    def _call(
        self,
        inputs: dict[str, list[Document]],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        """Prepare inputs, call combine docs, prepare outputs."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        docs = inputs[self.input_key]
        # Other keys are assumed to be needed for LLM prediction
        other_keys = {k: v for k, v in inputs.items() if k != self.input_key}
        output, extra_return_dict = self.combine_docs(
            docs,
            callbacks=_run_manager.get_child(),
            **other_keys,
        )
        extra_return_dict[self.output_key] = output
        return extra_return_dict

    async def _acall(
        self,
        inputs: dict[str, list[Document]],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        """Prepare inputs, call combine docs, prepare outputs."""
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        docs = inputs[self.input_key]
        # Other keys are assumed to be needed for LLM prediction
        other_keys = {k: v for k, v in inputs.items() if k != self.input_key}
        output, extra_return_dict = await self.acombine_docs(
            docs,
            callbacks=_run_manager.get_child(),
            **other_keys,
        )
        extra_return_dict[self.output_key] = output
        return extra_return_dict


@deprecated(
    since="0.2.7",
    alternative=(
        "example in API reference with more detail: "
        "https://api.python.langchain.com/en/latest/chains/langchain.chains.combine_documents.base.AnalyzeDocumentChain.html"
    ),
    removal="1.0",
)
class AnalyzeDocumentChain(Chain):
    """Chain that splits documents, then analyzes it in pieces.

    This chain is parameterized by a TextSplitter and a CombineDocumentsChain.
    This chain takes a single document as input, and then splits it up into chunks
    and then passes those chucks to the CombineDocumentsChain.

    This class is deprecated. See below for alternative implementations which
    supports async and streaming modes of operation.

    If the underlying combine documents chain takes one `input_documents` argument
    (e.g., chains generated by `load_summarize_chain`):

        ```python
        split_text = lambda x: text_splitter.create_documents([x])

        summarize_document_chain = split_text | chain
        ```

    If the underlying chain takes additional arguments (e.g., `load_qa_chain`, which
    takes an additional `question` argument), we can use the following:

        ```python
        from operator import itemgetter
        from langchain_core.runnables import RunnableLambda, RunnableParallel

        split_text = RunnableLambda(lambda x: text_splitter.create_documents([x]))
        summarize_document_chain = RunnableParallel(
            question=itemgetter("question"),
            input_documents=itemgetter("input_document") | split_text,
        ) | chain.pick("output_text")
        ```

    To additionally return the input parameters, as `AnalyzeDocumentChain` does,
    we can wrap this construction with `RunnablePassthrough`:

        ```python
        from operator import itemgetter
        from langchain_core.runnables import (
            RunnableLambda,
            RunnableParallel,
            RunnablePassthrough,
        )

        split_text = RunnableLambda(lambda x: text_splitter.create_documents([x]))
        summarize_document_chain = RunnablePassthrough.assign(
            output_text=RunnableParallel(
                question=itemgetter("question"),
                input_documents=itemgetter("input_document") | split_text,
            )
            | chain.pick("output_text")
        )
        ```
    """

    input_key: str = "input_document"
    text_splitter: TextSplitter = Field(default_factory=RecursiveCharacterTextSplitter)
    combine_docs_chain: BaseCombineDocumentsChain

    @property
    def input_keys(self) -> list[str]:
        """Expect input key."""
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        """Return output key."""
        return self.combine_docs_chain.output_keys

    @override
    def get_input_schema(
        self,
        config: RunnableConfig | None = None,
    ) -> type[BaseModel]:
        return create_model(
            "AnalyzeDocumentChain",
            **{self.input_key: (str, None)},
        )

    @override
    def get_output_schema(
        self,
        config: RunnableConfig | None = None,
    ) -> type[BaseModel]:
        return self.combine_docs_chain.get_output_schema(config)

    def _call(
        self,
        inputs: dict[str, str],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        """Split document into chunks and pass to CombineDocumentsChain."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        document = inputs[self.input_key]
        docs = self.text_splitter.create_documents([document])
        # Other keys are assumed to be needed for LLM prediction
        other_keys: dict = {k: v for k, v in inputs.items() if k != self.input_key}
        other_keys[self.combine_docs_chain.input_key] = docs
        return self.combine_docs_chain(
            other_keys,
            return_only_outputs=True,
            callbacks=_run_manager.get_child(),
        )
