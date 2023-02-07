"""Chain for chatting with a vector database."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

from langchain.chains.base import Chain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate
from langchain.vectorstores.base import VectorStore


def _get_chat_history(chat_history: List[Tuple[str, str]]) -> str:
    buffer = ""
    for human_s, ai_s in chat_history:
        human = "Human: " + human_s
        ai = "Assistant: " + ai_s
        buffer += "\n" + "\n".join([human, ai])
    return buffer


class ChatVectorDBChain(Chain, BaseModel):
    """Chain for chatting with a vector database."""

    vectorstore: VectorStore
    combine_docs_chain: BaseCombineDocumentsChain
    question_generator: LLMChain
    output_key: str = "answer"

    @property
    def _chain_type(self) -> str:
        return "chat-vector-db"

    @property
    def input_keys(self) -> List[str]:
        """Input keys."""
        return ["question", "chat_history"]

    @property
    def output_keys(self) -> List[str]:
        """Output keys."""
        return [self.output_key]

    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        vectorstore: VectorStore,
        condense_question_prompt: BasePromptTemplate = CONDENSE_QUESTION_PROMPT,
        qa_prompt: BasePromptTemplate = QA_PROMPT,
        chain_type: str = "stuff",
    ) -> ChatVectorDBChain:
        """Load chain from LLM."""
        doc_chain = load_qa_chain(
            llm,
            chain_type=chain_type,
            prompt=qa_prompt,
        )
        condense_question_chain = LLMChain(llm=llm, prompt=condense_question_prompt)
        return cls(
            vectorstore=vectorstore,
            combine_docs_chain=doc_chain,
            question_generator=condense_question_chain,
        )

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        question = inputs["question"]
        chat_history_str = _get_chat_history(inputs["chat_history"])
        if chat_history_str:
            new_question = self.question_generator.run(
                question=question, chat_history=chat_history_str
            )
        else:
            new_question = question
        docs = self.vectorstore.similarity_search(new_question, k=4)
        new_inputs = inputs.copy()
        new_inputs["question"] = new_question
        new_inputs["chat_history"] = chat_history_str
        answer, _ = self.combine_docs_chain.combine_docs(docs, **new_inputs)
        return {self.output_key: answer}
