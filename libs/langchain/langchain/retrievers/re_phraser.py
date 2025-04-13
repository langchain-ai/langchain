import logging

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)

# Default template
DEFAULT_TEMPLATE = """You are an assistant tasked with taking a natural language \
query from a user and converting it into a query for a vectorstore. \
In this process, you strip out information that is not relevant for \
the retrieval task. Here is the user query: {question}"""

# Default prompt
DEFAULT_QUERY_PROMPT = PromptTemplate.from_template(DEFAULT_TEMPLATE)


class RePhraseQueryRetriever(BaseRetriever):
    """Given a query, use an LLM to re-phrase it.
    Then, retrieve docs for the re-phrased query."""

    retriever: BaseRetriever
    llm_chain: Runnable

    @classmethod
    def from_llm(
        cls,
        retriever: BaseRetriever,
        llm: BaseLLM,
        prompt: BasePromptTemplate = DEFAULT_QUERY_PROMPT,
    ) -> "RePhraseQueryRetriever":
        """Initialize from llm using default template.

        The prompt used here expects a single input: `question`

        Args:
            retriever: retriever to query documents from
            llm: llm for query generation using DEFAULT_QUERY_PROMPT
            prompt: prompt template for query generation

        Returns:
            RePhraseQueryRetriever
        """
        llm_chain = prompt | llm | StrOutputParser()
        return cls(
            retriever=retriever,
            llm_chain=llm_chain,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Get relevant documents given a user question.

        Args:
            query: user question

        Returns:
            Relevant documents for re-phrased question
        """
        re_phrased_question = self.llm_chain.invoke(
            query, {"callbacks": run_manager.get_child()}
        )
        logger.info(f"Re-phrased question: {re_phrased_question}")
        docs = self.retriever.invoke(
            re_phrased_question, config={"callbacks": run_manager.get_child()}
        )
        return docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        raise NotImplementedError
