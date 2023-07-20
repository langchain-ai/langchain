import logging
from typing import List

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseRetriever, Document

logger = logging.getLogger(__name__)

# Default prompt
DEFAULT_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an assistant tasked with taking a 
    natural language query from a user and converting 
    it into a query for a vectorstore. In this process, 
    you strip out information that is not relevant for 
    the retrieval task. Here is the user query: {question} """,
)


class RePhraseQueryRetriever(BaseRetriever):

    """Given a user query, use an LLM to re-phrase it.
    Then, retrieve docs for re-phrased query."""

    retriever: BaseRetriever
    llm_chain: LLMChain

    @classmethod
    def from_llm(
        cls,
        retriever: BaseRetriever,
        llm: BaseLLM,
        prompt: PromptTemplate = DEFAULT_QUERY_PROMPT,
    ) -> "RePhraseQueryRetriever":
        """Initialize from llm using default template.

        Args:
            retriever: retriever to query documents from
            llm: llm for query generation using DEFAULT_QUERY_PROMPT
            prompt: prompt template for query generation

        Returns:
            RePhraseQueryRetriever
        """

        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(
            retriever=retriever,
            llm_chain=llm_chain,
        )

    def re_phase_question(
        self, question: str, run_manager: CallbackManagerForRetrieverRun
    ) -> str:
        """Generate queries based upon user input.

        Args:
            question: user question

        Returns:
            Re-phased user question
        """
        response = self.llm_chain(question, callbacks=run_manager.get_child())
        return response["text"]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Get relevated documents given a user question.

        Args:
            query: user question

        Returns:
            Relevant documents for re-phrased question
        """
        re_phrased_question = self.re_phase_question(query, run_manager)
        logger.info(f"Re-phrased question: {re_phrased_question}")
        docs = self.retriever.get_relevant_documents(
            re_phrased_question, callbacks=run_manager.get_child()
        )
        return docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        raise NotImplementedError
