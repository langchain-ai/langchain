"""Chain that combines documents by stuffing into context."""

from typing import Any, Optional

from langchain_core._api import deprecated
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import BasePromptTemplate, format_document
from langchain_core.runnables import Runnable, RunnablePassthrough
from pydantic import ConfigDict, Field, model_validator

from langchain.chains.combine_documents.base import (
    DEFAULT_DOCUMENT_PROMPT,
    DEFAULT_DOCUMENT_SEPARATOR,
    DOCUMENTS_KEY,
    BaseCombineDocumentsChain,
    _validate_prompt,
)
from langchain.chains.llm import LLMChain


def create_stuff_documents_chain(
    llm: LanguageModelLike,
    prompt: BasePromptTemplate,
    *,
    output_parser: Optional[BaseOutputParser] = None,
    document_prompt: Optional[BasePromptTemplate] = None,
    document_separator: str = DEFAULT_DOCUMENT_SEPARATOR,
    document_variable_name: str = DOCUMENTS_KEY,
) -> Runnable[dict[str, Any], Any]:
    """Create a chain for passing a list of Documents to a model.

    Args:
        llm: Language model.
        prompt: Prompt template. Must contain input variable "context" (override by
            setting document_variable), which will be used for passing in the formatted documents.
        output_parser: Output parser. Defaults to StrOutputParser.
        document_prompt: Prompt used for formatting each document into a string. Input
            variables can be "page_content" or any metadata keys that are in all
            documents. "page_content" will automatically retrieve the
            `Document.page_content`, and all other inputs variables will be
            automatically retrieved from the `Document.metadata` dictionary. Default to
            a prompt that only contains `Document.page_content`.
        document_separator: String separator to use between formatted document strings.
        document_variable_name: Variable name to use for the formatted documents in the prompt.
            Defaults to "context".

    Returns:
        An LCEL Runnable. The input is a dictionary that must have a "context" key that
        maps to a List[Document], and any other input variables expected in the prompt.
        The Runnable return type depends on output_parser used.

    Example:
        .. code-block:: python

            # pip install -U langchain langchain-community

            from langchain_community.chat_models import ChatOpenAI
            from langchain_core.documents import Document
            from langchain_core.prompts import ChatPromptTemplate
            from langchain.chains.combine_documents import create_stuff_documents_chain

            prompt = ChatPromptTemplate.from_messages(
                [("system", "What are everyone's favorite colors:\\n\\n{context}")]
            )
            llm = ChatOpenAI(model="gpt-3.5-turbo")
            chain = create_stuff_documents_chain(llm, prompt)

            docs = [
                Document(page_content="Jesse loves red but not yellow"),
                Document(page_content = "Jamal loves green but not as much as he loves orange")
            ]

            chain.invoke({"context": docs})
    """  # noqa: E501

    _validate_prompt(prompt, document_variable_name)
    _document_prompt = document_prompt or DEFAULT_DOCUMENT_PROMPT
    _output_parser = output_parser or StrOutputParser()

    def format_docs(inputs: dict) -> str:
        return document_separator.join(
            format_document(doc, _document_prompt)
            for doc in inputs[document_variable_name]
        )

    return (
        RunnablePassthrough.assign(**{document_variable_name: format_docs}).with_config(
            run_name="format_inputs"
        )
        | prompt
        | llm
        | _output_parser
    ).with_config(run_name="stuff_documents_chain")


@deprecated(
    since="0.2.13",
    removal="1.0",
    message=(
        "This class is deprecated. Use the `create_stuff_documents_chain` constructor "
        "instead. See migration guide here: "
        "https://python.langchain.com/docs/versions/migrating_chains/stuff_docs_chain/"  # noqa: E501
    ),
)
class StuffDocumentsChain(BaseCombineDocumentsChain):
    """Chain that combines documents by stuffing into context.

    This chain takes a list of documents and first combines them into a single string.
    It does this by formatting each document into a string with the `document_prompt`
    and then joining them together with `document_separator`. It then adds that new
    string to the inputs with the variable name set by `document_variable_name`.
    Those inputs are then passed to the `llm_chain`.

    Example:
        .. code-block:: python

            from langchain.chains import StuffDocumentsChain, LLMChain
            from langchain_core.prompts import PromptTemplate
            from langchain_community.llms import OpenAI

            # This controls how each document will be formatted. Specifically,
            # it will be passed to `format_document` - see that function for more
            # details.
            document_prompt = PromptTemplate(
                input_variables=["page_content"],
                template="{page_content}"
            )
            document_variable_name = "context"
            llm = OpenAI()
            # The prompt here should take as an input variable the
            # `document_variable_name`
            prompt = PromptTemplate.from_template(
                "Summarize this content: {context}"
            )
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            chain = StuffDocumentsChain(
                llm_chain=llm_chain,
                document_prompt=document_prompt,
                document_variable_name=document_variable_name
            )
    """

    llm_chain: LLMChain
    """LLM chain which is called with the formatted document string,
    along with any other inputs."""
    document_prompt: BasePromptTemplate = Field(
        default_factory=lambda: DEFAULT_DOCUMENT_PROMPT
    )
    """Prompt to use to format each document, gets passed to `format_document`."""
    document_variable_name: str
    """The variable name in the llm_chain to put the documents in.
    If only one variable in the llm_chain, this need not be provided."""
    document_separator: str = "\n\n"
    """The string with which to join the formatted documents"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def get_default_document_variable_name(cls, values: dict) -> Any:
        """Get default document variable name, if not provided.

        If only one variable is present in the llm_chain.prompt,
        we can infer that the formatted documents should be passed in
        with this variable name.
        """
        llm_chain_variables = values["llm_chain"].prompt.input_variables
        if "document_variable_name" not in values:
            if len(llm_chain_variables) == 1:
                values["document_variable_name"] = llm_chain_variables[0]
            else:
                raise ValueError(
                    "document_variable_name must be provided if there are "
                    "multiple llm_chain_variables"
                )
        else:
            if values["document_variable_name"] not in llm_chain_variables:
                raise ValueError(
                    f"document_variable_name {values['document_variable_name']} was "
                    f"not found in llm_chain input_variables: {llm_chain_variables}"
                )
        return values

    @property
    def input_keys(self) -> list[str]:
        extra_keys = [
            k for k in self.llm_chain.input_keys if k != self.document_variable_name
        ]
        return super().input_keys + extra_keys

    def _get_inputs(self, docs: list[Document], **kwargs: Any) -> dict:
        """Construct inputs from kwargs and docs.

        Format and then join all the documents together into one input with name
        `self.document_variable_name`. Also pluck any additional variables
        from **kwargs.

        Args:
            docs: List of documents to format and then join into single input
            **kwargs: additional inputs to chain, will pluck any other required
                arguments from here.

        Returns:
            dictionary of inputs to LLMChain
        """
        # Format each document according to the prompt
        doc_strings = [format_document(doc, self.document_prompt) for doc in docs]
        # Join the documents together to put them in the prompt.
        inputs = {
            k: v
            for k, v in kwargs.items()
            if k in self.llm_chain.prompt.input_variables
        }
        inputs[self.document_variable_name] = self.document_separator.join(doc_strings)
        return inputs

    def prompt_length(self, docs: list[Document], **kwargs: Any) -> Optional[int]:
        """Return the prompt length given the documents passed in.

        This can be used by a caller to determine whether passing in a list
        of documents would exceed a certain prompt length. This useful when
        trying to ensure that the size of a prompt remains below a certain
        context limit.

        Args:
            docs: List[Document], a list of documents to use to calculate the
                total prompt length.

        Returns:
            Returns None if the method does not depend on the prompt length,
            otherwise the length of the prompt in tokens.
        """
        inputs = self._get_inputs(docs, **kwargs)
        prompt = self.llm_chain.prompt.format(**inputs)
        return self.llm_chain._get_num_tokens(prompt)

    def combine_docs(
        self, docs: list[Document], callbacks: Callbacks = None, **kwargs: Any
    ) -> tuple[str, dict]:
        """Stuff all documents into one prompt and pass to LLM.

        Args:
            docs: List of documents to join together into one variable
            callbacks: Optional callbacks to pass along
            **kwargs: additional parameters to use to get inputs to LLMChain.

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        """
        inputs = self._get_inputs(docs, **kwargs)
        # Call predict on the LLM.
        return self.llm_chain.predict(callbacks=callbacks, **inputs), {}

    async def acombine_docs(
        self, docs: list[Document], callbacks: Callbacks = None, **kwargs: Any
    ) -> tuple[str, dict]:
        """Async stuff all documents into one prompt and pass to LLM.

        Args:
            docs: List of documents to join together into one variable
            callbacks: Optional callbacks to pass along
            **kwargs: additional parameters to use to get inputs to LLMChain.

        Returns:
            The first element returned is the single string output. The second
            element returned is a dictionary of other keys to return.
        """
        inputs = self._get_inputs(docs, **kwargs)
        # Call predict on the LLM.
        return await self.llm_chain.apredict(callbacks=callbacks, **inputs), {}

    @property
    def _chain_type(self) -> str:
        return "stuff_documents_chain"
