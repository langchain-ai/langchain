"""Chain for question-answering against a vector database."""
from typing import Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.vector_db_qa.prompt import PROMPT
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.vectorstores.base import VectorStore


class VectorDBQA(Chain, BaseModel):
    """Chain for question-answering against a vector database.

    Example:
        .. code-block:: python

            from langchain import OpenAI, VectorDBQA
            from langchain.faiss import FAISS
            vectordb = FAISS(...)
            vectordbQA = VectorDBQA(llm=OpenAI(), vector_db=vectordb)

    """

    llm: LLM
    """LLM wrapper to use."""
    vectorstore: VectorStore
    """Vector Database to connect to."""
    k: int = 4
    """Number of documents to query for."""
    prompt: PromptTemplate = PROMPT
    """Prompt to use when questioning the documents."""
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        question = inputs[self.input_key]
        llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
        docs = self.vectorstore.similarity_search(question, k=self.k)
        contexts = []
        for j, doc in enumerate(docs):
            contexts.append(f"Context {j}:\n{doc.page_content}")
        # TODO: handle cases where this context is too long.
        answer = llm_chain.predict(question=question, context="\n\n".join(contexts))
        return {self.output_key: answer}
