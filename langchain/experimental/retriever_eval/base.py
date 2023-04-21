""""""
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from pydantic import BaseModel, Field

from langchain import OpenAI
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.question_answering import load_qa_chain
from langchain.evaluation.qa import QAEvalChain
from langchain.schema import BaseRetriever, Document


class TestRetriever(BaseRetriever, BaseModel, ABC):
    """Retriever that can also ingest new documents."""

    identifying_params: dict

    def add_documents(self, docs: List[Document], can_edit: bool = True) -> None:
        """"""
        if can_edit:
            docs = self._transform_documents(docs)
        self._insert_documents(docs)

    def _transform_documents(self, docs: List[Document]) -> List[Document]:
        """"""
        return docs

    @abstractmethod
    def _insert_documents(self, docs: List[Document]) -> None:
        """"""

    def cleanup(self) -> None:
        pass

    @property
    def name(self) -> str:
        return str(self.identifying_params)


class RetrieverTestCase(BaseModel, ABC):
    """"""

    name: str
    query: str
    docs: List[Document]
    can_edit_docs: bool = True

    @classmethod
    def from_config(cls, **kwargs: Any) -> "RetrieverTestCase":
        """"""
        return cls(**kwargs)

    @abstractmethod
    def check_retrieved_docs(self, retrieved_docs: List[Document]) -> bool:
        """"""

    def run(self, retriever: TestRetriever) -> Tuple[bool, dict]:
        retriever.add_documents(self.docs, can_edit=self.can_edit_docs)
        retrieved_docs = retriever.get_relevant_documents(self.query)
        passed = self.check_retrieved_docs(retrieved_docs)
        extra_dict = {"retrieved_docs": retrieved_docs}
        retriever.cleanup()
        return passed, extra_dict


class QAEvalChainTestCase(RetrieverTestCase):
    """"""

    gold_standard_answer: str
    qa_chain: BaseCombineDocumentsChain = Field(
        default_factory=lambda: load_qa_chain(OpenAI(temperature=0))
    )
    qa_eval_chain: QAEvalChain = Field(
        default_factory=lambda: QAEvalChain.from_llm(OpenAI(temperature=0))
    )

    def check_retrieved_docs(self, retrieved_docs: List[Document]) -> bool:
        qa_response = self.qa_chain(
            {"input_documents": retrieved_docs, "question": self.query}
        )
        qa_response["answer"] = self.gold_standard_answer
        return self.qa_eval_chain.predict_and_parse(qa_response)


class ExpectedSubstringsTestCase(RetrieverTestCase):
    expected_substrings: List[str]

    def check_retrieved_docs(self, retrieved_docs: List[Document]) -> bool:
        """"""
        all_text = "\n".join([d.page_content for d in retrieved_docs])
        for substring in self.expected_substrings:
            if substring not in all_text:
                return False
        return True
