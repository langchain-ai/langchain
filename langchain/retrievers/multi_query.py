from typing import List

from pydantic import BaseModel, Field

from langchain import LLMChain
from langchain.llms import BaseLLM
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document


class LineList(BaseModel):
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


class MultiQueryRetriever:

    """Given a user query, use an LLM to write a set of `num_queries` queries.
    Retrieve docs for each query. Rake the unique union of all retrieved docs."""

    # Prompt
    DEFAULT_QUERY_PROMPT = PromptTemplate(
        input_variables=["question", "num_queries"],
        template="""You are an AI language model assistant. Your task is 
        to generate {num_queries} different versions of the given user 
        question to retrieve relevant documents from a vector  database. 
        By generating multiple perspectives on the user question, 
        your goal is to help the user overcome some of the limitations 
        of distance-based similarity search. Provide these alternative 
        questions seperated by newlines. Original question: {question}""",
    )

    def __init__(
        self,
        retriever: BaseRetriever,
        num_queries: int,
        llm_chain: LLMChain,
        verbose: bool = True,
    ) -> None:
        """Initialize MultiQueryRetriever.

        Args:
            retriever: retriever to query documents from
            num_queries: number of queries for the LLM to generate
            llm_chain: llm_chain for query generation
            verbose: show the queries that we generated to the user

        Returns:
            MultiQueryRetriever
        """
        self.retriever = retriever
        self.num_queries = num_queries
        self.llm_chain = llm_chain
        self.verbose = verbose

    @classmethod
    def from_llm(
        cls, retriever: BaseRetriever, num_queries: int, llm: BaseLLM
    ) -> "MultiQueryRetriever":
        """Initialize from llm using default template.

        Args:
            retriever: retriever to query documents from
            num_queries: number of queries for the LLM to generate
            llm: llm for query generation using DEFAULT_QUERY_PROMPT

        Returns:
            MultiQueryRetriever
        """
        llm_chain = LLMChain(llm=llm, prompt=cls.DEFAULT_QUERY_PROMPT)
        return cls(retriever=retriever, num_queries=num_queries, llm_chain=llm_chain)

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

    def generate_queries(self, question: str) -> List[str]:
        """Generate queries based upon user input.

        Args:
            question: user query

        Returns:
            List of LLM generated queries that are similar to the user input
        """
        response = self.llm_chain.apply(
            [{"question": question, "num_queries": self.num_queries}]
        )
        # Get the response text
        response_text = response[0]["text"]
        # Parse the response
        output_parser = LineListOutputParser()
        parsed_output = output_parser.parse(response_text)
        if self.verbose:
            print(f"Generated queries: {parsed_output.lines}")
        return parsed_output.lines

    def retrieve_documents(self, queries: List[str]) -> List[Document]:
        """Run all LLM generated queries.

        Args:
            queries: query list

        Returns:
            List of retrived Documents
        """
        documents = []
        for query in queries:
            docs = self.retriever.get_relevant_documents(
                query
            )  # assuming this method exists
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
        unique_documents_dict = {doc.page_content: doc for doc in documents}
        unique_documents = list(unique_documents_dict.values())
        return unique_documents
