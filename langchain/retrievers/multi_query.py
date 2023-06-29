import logging
from typing import List

from pydantic import BaseModel, Field

from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseRetriever, Document

logger = logging.getLogger(__name__)


class LineList(BaseModel):
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


# Default prompt
DEFAULT_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is 
    to generate 3 different versions of the given user 
    question to retrieve relevant documents from a vector  database. 
    By generating multiple perspectives on the user question, 
    your goal is to help the user overcome some of the limitations 
    of distance-based similarity search. Provide these alternative 
    questions seperated by newlines. Original question: {question}""",
)


class MultiQueryRetriever(BaseRetriever):

    """Given a user query, use an LLM to write a set of queries.
    Retrieve docs for each query. Rake the unique union of all retrieved docs."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm_chain: LLMChain,
        verbose: bool = True,
        parser_key: str = "lines",
    ) -> None:
        """Initialize MultiQueryRetriever.

        Args:
            retriever: retriever to query documents from
            llm_chain: llm_chain for query generation
            verbose: show the queries that we generated to the user
            parser_key: attribute name for the parsed output

        Returns:
            MultiQueryRetriever
        """
        self.retriever = retriever
        self.llm_chain = llm_chain
        self.verbose = verbose
        self.parser_key = parser_key

    @classmethod
    def from_llm(
        cls,
        retriever: BaseRetriever,
        llm: BaseLLM,
        prompt: PromptTemplate = DEFAULT_QUERY_PROMPT,
        parser_key: str = "lines",
    ) -> "MultiQueryRetriever":
        """Initialize from llm using default template.

        Args:
            retriever: retriever to query documents from
            llm: llm for query generation using DEFAULT_QUERY_PROMPT

        Returns:
            MultiQueryRetriever
        """
        output_parser = LineListOutputParser()
        llm_chain = LLMChain(llm=llm, prompt=prompt, output_parser=output_parser)
        return cls(
            retriever=retriever,
            llm_chain=llm_chain,
            parser_key=parser_key,
        )

    def get_relevant_documents(self, question: str) -> List[Document]:
        """Get relevated documents given a user query.

        Args:
            question: user query

        Returns:
            Unique union of relevant documents from all generated queries
        """
        queries = self.generate_queries(question)
        documents = self.retrieve_documents(queries)
        unique_documents = self.unique_union(documents)
        return unique_documents

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError

    def generate_queries(self, question: str) -> List[str]:
        """Generate queries based upon user input.

        Args:
            question: user query

        Returns:
            List of LLM generated queries that are similar to the user input
        """
        response = self.llm_chain({"question": question})
        lines = getattr(response["text"], self.parser_key, [])
        if self.verbose:
            logger.info(f"Generated queries: {lines}")
        return lines

    def retrieve_documents(self, queries: List[str]) -> List[Document]:
        """Run all LLM generated queries.

        Args:
            queries: query list

        Returns:
            List of retrived Documents
        """
        documents = []
        for query in queries:
            docs = self.retriever.get_relevant_documents(query)
            documents.extend(docs)
        return documents

    def unique_union(self, documents: List[Document]) -> List[Document]:
        """Get uniqe Documents.

        Args:
            documents: List of retrived Documents

        Returns:
            List of unique retrived Documents
        """
        # Create a dictionary with page_content as keys to remove duplicates
        # TODO: Add Document ID property (e.g., UUID)
        unique_documents_dict = {
            (doc.page_content, tuple(sorted(doc.metadata.items()))): doc
            for doc in documents
        }

        unique_documents = list(unique_documents_dict.values())
        return unique_documents
