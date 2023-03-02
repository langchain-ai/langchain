"""Chain for chatting with a vector database."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chat.base import BaseChatChain
from langchain.chat.question_answering import QAChain
from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import BaseLLM
from langchain.memory.utils import get_buffer_string
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import ChatMessage
from langchain.vectorstores.base import VectorStore


class ChatVectorDBChain(BaseChatChain, BaseModel):
    """Chain for chatting with a vector database."""

    vectorstore: VectorStore
    qa_chain: QAChain
    question_generator: LLMChain
    output_key: str = "answer"
    return_source_documents: bool = False
    top_k_docs_for_context: int = 4
    """Return the source documents."""

    @property
    def _chain_type(self) -> str:
        return "chat-vector-db"

    @property
    def input_keys(self) -> List[str]:
        """Input keys."""
        return ["question", "chat_history"]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.

        :meta private:
        """
        _output_keys = [self.output_key]
        if self.return_source_documents:
            _output_keys = _output_keys + ["source_documents"]
        return _output_keys

    @classmethod
    def from_llm(
        cls,
        *,
        llm: BaseLLM,
        model: BaseChatModel,
        vectorstore: VectorStore,
        starter_messages: Optional[List[ChatMessage]] = None,
        condense_question_prompt: BasePromptTemplate = CONDENSE_QUESTION_PROMPT,
        **kwargs: Any,
    ) -> ChatVectorDBChain:
        """Load chain from LLM."""
        qa_chain = QAChain.from_model(model, starter_messages=starter_messages)
        condense_question_chain = LLMChain(llm=llm, prompt=condense_question_prompt)
        return cls(
            vectorstore=vectorstore,
            qa_chain=qa_chain,
            question_generator=condense_question_chain,
            **kwargs,
        )

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        chat_history_str = get_buffer_string(inputs["chat_history"])
        vectordbkwargs = inputs.get("vectordbkwargs", {})
        if chat_history_str:
            new_question = self.question_generator.run(
                question=question, chat_history=chat_history_str
            )
        else:
            new_question = question
        docs = self.vectorstore.similarity_search(
            new_question, k=self.top_k_docs_for_context, **vectordbkwargs
        )
        args = {
            self.qa_chain.documents_key: docs,
            self.qa_chain.question_key: new_question,
        }

        result = self.qa_chain(args)
        answer = result[self.qa_chain.output_key]
        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}
