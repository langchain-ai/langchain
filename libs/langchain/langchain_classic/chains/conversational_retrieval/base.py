"""Chain for chatting with a vector database."""

from __future__ import annotations

import inspect
import warnings
from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core._api import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import BasePromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import override

from langchain_classic.chains.base import Chain
from langchain_classic.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_classic.chains.conversational_retrieval.prompts import (
    CONDENSE_QUESTION_PROMPT,
)
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.question_answering import load_qa_chain

# Depending on the memory type and configuration, the chat history format may differ.
# This needs to be consolidated.
CHAT_TURN_TYPE = tuple[str, str] | BaseMessage


_ROLE_MAP = {"human": "Human: ", "ai": "Assistant: "}


def _get_chat_history(chat_history: list[CHAT_TURN_TYPE]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        if isinstance(dialogue_turn, BaseMessage):
            if len(dialogue_turn.content) > 0:
                role_prefix = _ROLE_MAP.get(
                    dialogue_turn.type,
                    f"{dialogue_turn.type}: ",
                )
                buffer += f"\n{role_prefix}{dialogue_turn.content}"
        elif isinstance(dialogue_turn, tuple):
            human = "Human: " + dialogue_turn[0]
            ai = "Assistant: " + dialogue_turn[1]
            buffer += f"\n{human}\n{ai}"
        else:
            msg = (  # type: ignore[unreachable]
                f"Unsupported chat history format: {type(dialogue_turn)}."
                f" Full chat history: {chat_history} "
            )
            raise ValueError(msg)  # noqa: TRY004
    return buffer


class InputType(BaseModel):
    """Input type for ConversationalRetrievalChain."""

    question: str
    """The question to answer."""
    chat_history: list[CHAT_TURN_TYPE] = Field(default_factory=list)
    """The chat history to use for retrieval."""


class BaseConversationalRetrievalChain(Chain):
    """Chain for chatting with an index."""

    combine_docs_chain: BaseCombineDocumentsChain
    """The chain used to combine any retrieved documents."""
    question_generator: LLMChain
    """The chain used to generate a new question for the sake of retrieval.
    This chain will take in the current question (with variable `question`)
    and any chat history (with variable `chat_history`) and will produce
    a new standalone question to be used later on."""
    output_key: str = "answer"
    """The output key to return the final answer of this chain in."""
    rephrase_question: bool = True
    """Whether or not to pass the new generated question to the combine_docs_chain.
    If `True`, will pass the new generated question along.
    If `False`, will only use the new generated question for retrieval and pass the
    original question along to the combine_docs_chain."""
    return_source_documents: bool = False
    """Return the retrieved source documents as part of the final result."""
    return_generated_question: bool = False
    """Return the generated question as part of the final result."""
    get_chat_history: Callable[[list[CHAT_TURN_TYPE]], str] | None = None
    """An optional function to get a string of the chat history.
    If `None` is provided, will use a default."""
    response_if_no_docs_found: str | None = None
    """If specified, the chain will return a fixed response if no docs
    are found for the question. """

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @property
    def input_keys(self) -> list[str]:
        """Input keys."""
        return ["question", "chat_history"]

    @override
    def get_input_schema(
        self,
        config: RunnableConfig | None = None,
    ) -> type[BaseModel]:
        return InputType

    @property
    def output_keys(self) -> list[str]:
        """Return the output keys.

        :meta private:
        """
        _output_keys = [self.output_key]
        if self.return_source_documents:
            _output_keys = [*_output_keys, "source_documents"]
        if self.return_generated_question:
            _output_keys = [*_output_keys, "generated_question"]
        return _output_keys

    @abstractmethod
    def _get_docs(
        self,
        question: str,
        inputs: dict[str, Any],
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> list[Document]:
        """Get docs."""

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs["question"]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])

        if chat_history_str:
            callbacks = _run_manager.get_child()
            new_question = self.question_generator.run(
                question=question,
                chat_history=chat_history_str,
                callbacks=callbacks,
            )
        else:
            new_question = question
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._get_docs).parameters
        )
        if accepts_run_manager:
            docs = self._get_docs(new_question, inputs, run_manager=_run_manager)
        else:
            docs = self._get_docs(new_question, inputs)  # type: ignore[call-arg]
        output: dict[str, Any] = {}
        if self.response_if_no_docs_found is not None and len(docs) == 0:
            output[self.output_key] = self.response_if_no_docs_found
        else:
            new_inputs = inputs.copy()
            if self.rephrase_question:
                new_inputs["question"] = new_question
            new_inputs["chat_history"] = chat_history_str
            answer = self.combine_docs_chain.run(
                input_documents=docs,
                callbacks=_run_manager.get_child(),
                **new_inputs,
            )
            output[self.output_key] = answer

        if self.return_source_documents:
            output["source_documents"] = docs
        if self.return_generated_question:
            output["generated_question"] = new_question
        return output

    @abstractmethod
    async def _aget_docs(
        self,
        question: str,
        inputs: dict[str, Any],
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> list[Document]:
        """Get docs."""

    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        question = inputs["question"]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])
        if chat_history_str:
            callbacks = _run_manager.get_child()
            new_question = await self.question_generator.arun(
                question=question,
                chat_history=chat_history_str,
                callbacks=callbacks,
            )
        else:
            new_question = question
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._aget_docs).parameters
        )
        if accepts_run_manager:
            docs = await self._aget_docs(new_question, inputs, run_manager=_run_manager)
        else:
            docs = await self._aget_docs(new_question, inputs)  # type: ignore[call-arg]

        output: dict[str, Any] = {}
        if self.response_if_no_docs_found is not None and len(docs) == 0:
            output[self.output_key] = self.response_if_no_docs_found
        else:
            new_inputs = inputs.copy()
            if self.rephrase_question:
                new_inputs["question"] = new_question
            new_inputs["chat_history"] = chat_history_str
            answer = await self.combine_docs_chain.arun(
                input_documents=docs,
                callbacks=_run_manager.get_child(),
                **new_inputs,
            )
            output[self.output_key] = answer

        if self.return_source_documents:
            output["source_documents"] = docs
        if self.return_generated_question:
            output["generated_question"] = new_question
        return output

    @override
    def save(self, file_path: Path | str) -> None:
        if self.get_chat_history:
            msg = "Chain not saveable when `get_chat_history` is not None."
            raise ValueError(msg)
        super().save(file_path)


@deprecated(
    since="0.1.17",
    alternative=(
        "create_history_aware_retriever together with create_retrieval_chain "
        "(see example in docstring)"
    ),
    removal="1.0",
)
class ConversationalRetrievalChain(BaseConversationalRetrievalChain):
    r"""Chain for having a conversation based on retrieved documents.

    This class is deprecated. See below for an example implementation using
    `create_retrieval_chain`. Additional walkthroughs can be found at
    https://python.langchain.com/docs/use_cases/question_answering/chat_history

    ```python
    from langchain_classic.chains import (
        create_history_aware_retriever,
        create_retrieval_chain,
    )
    from langchain_classic.chains.combine_documents import (
        create_stuff_documents_chain,
    )
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_openai import ChatOpenAI

    retriever = ...  # Your retriever

    model = ChatOpenAI()

    # Contextualize question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )

    # Answer question
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Use three sentences maximum and keep the answer "
        "concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    # Below we use create_stuff_documents_chain to feed all retrieved context
    # into the LLM. Note that we can also use StuffDocumentsChain and other
    # instances of BaseCombineDocumentsChain.
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Usage:
    chat_history = []  # Collect chat history here (a sequence of messages)
    rag_chain.invoke({"input": query, "chat_history": chat_history})
    ```

    This chain takes in chat history (a list of messages) and new questions,
    and then returns an answer to that question.
    The algorithm for this chain consists of three parts:

    1. Use the chat history and the new question to create a "standalone question".
    This is done so that this question can be passed into the retrieval step to fetch
    relevant documents. If only the new question was passed in, then relevant context
    may be lacking. If the whole conversation was passed into retrieval, there may
    be unnecessary information there that would distract from retrieval.

    2. This new question is passed to the retriever and relevant documents are
    returned.

    3. The retrieved documents are passed to an LLM along with either the new question
    (default behavior) or the original question and chat history to generate a final
    response.

    Example:
        ```python
        from langchain_classic.chains import (
            StuffDocumentsChain,
            LLMChain,
            ConversationalRetrievalChain,
        )
        from langchain_core.prompts import PromptTemplate
        from langchain_community.llms import OpenAI

        combine_docs_chain = StuffDocumentsChain(...)
        vectorstore = ...
        retriever = vectorstore.as_retriever()

        # This controls how the standalone question is generated.
        # Should take `chat_history` and `question` as input variables.
        template = (
            "Combine the chat history and follow up question into "
            "a standalone question. Chat History: {chat_history}"
            "Follow up question: {question}"
        )
        prompt = PromptTemplate.from_template(template)
        model = OpenAI()
        question_generator_chain = LLMChain(llm=model, prompt=prompt)
        chain = ConversationalRetrievalChain(
            combine_docs_chain=combine_docs_chain,
            retriever=retriever,
            question_generator=question_generator_chain,
        )
        ```
    """

    retriever: BaseRetriever
    """Retriever to use to fetch documents."""
    max_tokens_limit: int | None = None
    """If set, enforces that the documents returned are less than this limit.

    This is only enforced if `combine_docs_chain` is of type StuffDocumentsChain.
    """

    def _reduce_tokens_below_limit(self, docs: list[Document]) -> list[Document]:
        num_docs = len(docs)

        if self.max_tokens_limit and isinstance(
            self.combine_docs_chain,
            StuffDocumentsChain,
        ):
            tokens = [
                self.combine_docs_chain.llm_chain._get_num_tokens(doc.page_content)  # noqa: SLF001
                for doc in docs
            ]
            token_count = sum(tokens[:num_docs])
            while token_count > self.max_tokens_limit:
                num_docs -= 1
                token_count -= tokens[num_docs]

        return docs[:num_docs]

    @override
    def _get_docs(
        self,
        question: str,
        inputs: dict[str, Any],
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> list[Document]:
        """Get docs."""
        docs = self.retriever.invoke(
            question,
            config={"callbacks": run_manager.get_child()},
        )
        return self._reduce_tokens_below_limit(docs)

    @override
    async def _aget_docs(
        self,
        question: str,
        inputs: dict[str, Any],
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> list[Document]:
        """Get docs."""
        docs = await self.retriever.ainvoke(
            question,
            config={"callbacks": run_manager.get_child()},
        )
        return self._reduce_tokens_below_limit(docs)

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        retriever: BaseRetriever,
        condense_question_prompt: BasePromptTemplate = CONDENSE_QUESTION_PROMPT,
        chain_type: str = "stuff",
        verbose: bool = False,  # noqa: FBT001,FBT002
        condense_question_llm: BaseLanguageModel | None = None,
        combine_docs_chain_kwargs: dict | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> BaseConversationalRetrievalChain:
        """Convenience method to load chain from LLM and retriever.

        This provides some logic to create the `question_generator` chain
        as well as the combine_docs_chain.

        Args:
            llm: The default language model to use at every part of this chain
                (eg in both the question generation and the answering)
            retriever: The retriever to use to fetch relevant documents from.
            condense_question_prompt: The prompt to use to condense the chat history
                and new question into a standalone question.
            chain_type: The chain type to use to create the combine_docs_chain, will
                be sent to `load_qa_chain`.
            verbose: Verbosity flag for logging to stdout.
            condense_question_llm: The language model to use for condensing the chat
                history and new question into a standalone question. If none is
                provided, will default to `llm`.
            combine_docs_chain_kwargs: Parameters to pass as kwargs to `load_qa_chain`
                when constructing the combine_docs_chain.
            callbacks: Callbacks to pass to all subchains.
            kwargs: Additional parameters to pass when initializing
                ConversationalRetrievalChain
        """
        combine_docs_chain_kwargs = combine_docs_chain_kwargs or {}
        doc_chain = load_qa_chain(
            llm,
            chain_type=chain_type,
            verbose=verbose,
            callbacks=callbacks,
            **combine_docs_chain_kwargs,
        )

        _llm = condense_question_llm or llm
        condense_question_chain = LLMChain(
            llm=_llm,
            prompt=condense_question_prompt,
            verbose=verbose,
            callbacks=callbacks,
        )
        return cls(
            retriever=retriever,
            combine_docs_chain=doc_chain,
            question_generator=condense_question_chain,
            callbacks=callbacks,
            **kwargs,
        )


class ChatVectorDBChain(BaseConversationalRetrievalChain):
    """Chain for chatting with a vector database."""

    vectorstore: VectorStore = Field(alias="vectorstore")
    top_k_docs_for_context: int = 4
    search_kwargs: dict = Field(default_factory=dict)

    @property
    def _chain_type(self) -> str:
        return "chat-vector-db"

    @model_validator(mode="before")
    @classmethod
    def _raise_deprecation(cls, values: dict) -> Any:
        warnings.warn(
            "`ChatVectorDBChain` is deprecated - "
            "please use `from langchain_classic.chains import "
            "ConversationalRetrievalChain`",
            stacklevel=4,
        )
        return values

    @override
    def _get_docs(
        self,
        question: str,
        inputs: dict[str, Any],
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> list[Document]:
        """Get docs."""
        vectordbkwargs = inputs.get("vectordbkwargs", {})
        full_kwargs = {**self.search_kwargs, **vectordbkwargs}
        return self.vectorstore.similarity_search(
            question,
            k=self.top_k_docs_for_context,
            **full_kwargs,
        )

    async def _aget_docs(
        self,
        question: str,
        inputs: dict[str, Any],
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> list[Document]:
        """Get docs."""
        msg = "ChatVectorDBChain does not support async"
        raise NotImplementedError(msg)

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        vectorstore: VectorStore,
        condense_question_prompt: BasePromptTemplate = CONDENSE_QUESTION_PROMPT,
        chain_type: str = "stuff",
        combine_docs_chain_kwargs: dict | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> BaseConversationalRetrievalChain:
        """Load chain from LLM."""
        combine_docs_chain_kwargs = combine_docs_chain_kwargs or {}
        doc_chain = load_qa_chain(
            llm,
            chain_type=chain_type,
            callbacks=callbacks,
            **combine_docs_chain_kwargs,
        )
        condense_question_chain = LLMChain(
            llm=llm,
            prompt=condense_question_prompt,
            callbacks=callbacks,
        )
        return cls(
            vectorstore=vectorstore,
            combine_docs_chain=doc_chain,
            question_generator=condense_question_chain,
            callbacks=callbacks,
            **kwargs,
        )
