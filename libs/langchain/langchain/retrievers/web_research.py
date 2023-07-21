import logging
import os
import re
from typing import List, Union

from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.chains import LLMChain
from langchain.chat_models.openai import ChatOpenAI
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
    llm: BaseLLM = Field(..., description="Language model for generating questions")
    llm: Union[BaseLLM, ChatOpenAI] = Field(
        ..., description="Language model for generating questions"
    )
    GOOGLE_CSE_ID: str = Field(..., description="Google Custom Search Engine ID")
    GOOGLE_API_KEY: str = Field(..., description="Google API Key")
    search_prompt: PromptTemplate = Field(
        DEFAULT_SEARCH_PROMPT, description="Search Prompt Template"
    )
    max_splits_per_doc: int = Field(100, description="Maximum splits per document")

    def search_tool(self, query: str, num_pages: int = 1):
        """Google search for up to 3 queries."""
        try:
            os.environ["GOOGLE_CSE_ID"] = self.GOOGLE_CSE_ID
            os.environ["GOOGLE_API_KEY"] = self.GOOGLE_API_KEY
            search = GoogleSearchAPIWrapper()
        except Exception as e:
            print(f"Error: {str(e)}")
        result = search.results(query, num_pages)
        return result if isinstance(result, list) else [result]

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
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=self.search_prompt,
            output_parser=QuestionListOutputParser(),
        )
        result = llm_chain({"question": query})
        logger.info(f"Questions for Google Search (raw): {result}")
        questions = getattr(result["text"], "lines", [])
        logger.info(f"Questions for Google Search: {questions}")

        # Get urls
        logger.info("Searching for relevat urls ...")
        urls_to_look = []
        for query in questions:
            # Google search
            search_results = self.search_tool(query)
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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=50
        )
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
