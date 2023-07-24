import logging
import re
from typing import List

from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.chains import LLMChain
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.llms.base import BaseLLM
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.vectorstores.base import VectorStore

logger = logging.getLogger(__name__)


class SearchQueries(BaseModel):
    """Search queries to run to research for the user's goal."""

    queries: List[str] = Field(
        ..., description="List of search queries to look up on Google"
    )


DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""<<SYS>> \n You are a web research assistant to help users
    answer questions. Answer using a numeric list. Do not include any extra
    test. \n <</SYS>> \n\n [INST] Given a user input search query, 
    generate a numbered list of five search queries to run to help answer their 
    question: \n\n {question} [/INST]""",
)


class LineList(BaseModel):
    """List of questions."""

    lines: List[str] = Field(description="Questions")


class QuestionListOutputParser(PydanticOutputParser):
    """Output parser for a list of numbered questions."""

    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = re.findall(r"\d+\..*?\n", text)
        return LineList(lines=lines)


class WebResearchRetriever(BaseRetriever):
    # Inputs
    vectorstore: VectorStore = Field(
        ..., description="Vector store for handling document embeddings"
    )
    llm_chain: LLMChain
    search: GoogleSearchAPIWrapper = Field(..., description="Google Search API Wrapper")
    search_prompt: PromptTemplate = Field(
        DEFAULT_SEARCH_PROMPT, description="Search Prompt Template"
    )
    max_splits_per_doc: int = Field(100, description="Maximum splits per document")
    num_search_results: int = Field(1, description="Number of pages per Google search")
    text_splitter: RecursiveCharacterTextSplitter = Field(
        RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50),
        description="Text splitter for splitting web pages into chunks",
    )

    @classmethod
    def from_llm(
        cls,
        vectorstore: VectorStore,
        llm: BaseLLM,
        search: GoogleSearchAPIWrapper,
        search_prompt: PromptTemplate = DEFAULT_SEARCH_PROMPT,
        max_splits_per_doc: int = 100,
        num_search_results: int = 1,
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=50
        ),
    ) -> "WebResearchRetriever":
        """Initialize from llm using default template.

        Args:
            search: GoogleSearchAPIWrapper
            llm: llm for search question generation using DEFAULT_SEARCH_PROMPT
            search_prompt: prompt to generating search questions
            max_splits_per_doc: Maximum splits per document to keep
            num_search_results: Number of pages per Google search
            text_splitter: Text splitter for splitting web pages into chunks

        Returns:
            WebResearchRetriever
        """
        llm_chain = LLMChain(
            llm=llm,
            prompt=search_prompt,
            output_parser=QuestionListOutputParser(),
        )
        return cls(
            vectorstore=vectorstore,
            llm_chain=llm_chain,
            search=search,
            search_prompt=search_prompt,
            max_splits_per_doc=max_splits_per_doc,
            num_search_results=num_search_results,
            text_splitter=text_splitter,
        )

    def search_tool(self, query: str, num_search_results: int = 1) -> List[dict]:
        """Returns num_serch_results pages per Google search."""
        try:
            result = self.search.results(query, num_search_results)
        except Exception as e:
            raise Exception(f"Error: {str(e)}")
        return result

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Search Google for documents related to the query input.

        Args:
            query: user query

        Returns:
            Relevant documents from all various urls.
        """

        # Get search questions
        logger.info("Generating questions for Google Search ...")
        result = self.llm_chain({"question": query})
        logger.info(f"Questions for Google Search (raw): {result}")
        questions = getattr(result["text"], "lines", [])
        logger.info(f"Questions for Google Search: {questions}")

        # Get urls
        logger.info("Searching for relevat urls ...")
        urls_to_look = []
        for query in questions:
            # Google search
            search_results = self.search_tool(query, self.num_search_results)
            logger.info("Searching for relevat urls ...")
            logger.info(f"Search results: {search_results}")
            for res in search_results:
                urls_to_look.append(res["link"])

        # Load HTML to text
        urls = set(urls_to_look)
        logger.info(f"URLs to load: {urls}")
        loader = AsyncHtmlLoader(list(urls))
        html2text = Html2TextTransformer()

        # Proect against very large documents
        # This can use rate limit w/ embedding
        logger.info("Grabbing most relevant splits from urls ...")
        filtered_splits = []
        text_splitter = self.text_splitter
        for doc in html2text.transform_documents(loader.load()):
            doc_splits = text_splitter.split_documents([doc])
            if len(doc_splits) > self.max_splits_per_doc:
                logger.info(
                    f"Document {doc.metadata} has too many splits ({len(doc_splits)}), "
                    f"keeping only the first {self.max_splits_per_doc}"
                )
                doc_splits = doc_splits[: self.max_splits_per_doc]
            filtered_splits.extend(doc_splits)
        self.vectorstore.add_documents(filtered_splits)

        # Search for relevant splits
        docs = []
        for query in questions:
            docs.extend(self.vectorstore.similarity_search(query))

        # Get unique docs
        unique_documents_dict = {
            (doc.page_content, tuple(sorted(doc.metadata.items()))): doc for doc in docs
        }
        unique_documents = list(unique_documents_dict.values())
        return unique_documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        raise NotImplementedError
