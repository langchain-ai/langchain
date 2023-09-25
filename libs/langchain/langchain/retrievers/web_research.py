import logging
import re
import ssl
from typing import List, Optional
from urllib.request import Request, urlopen

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.chains import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_loaders.async_pdf import AsyncPdfLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.llms import LlamaCpp
from langchain.llms.base import BaseLLM
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import BaseRetriever, Document
from langchain.schema.vectorstore import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utils.http import get_request_headers

logger = logging.getLogger(__name__)


class SearchQueries(BaseModel):
    """Search queries to research for the user's goal."""

    queries: List[str] = Field(
        ..., description="List of search queries to look up on Google"
    )


DEFAULT_LLAMA_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""<<SYS>> \n You are an assistant tasked with improving Google search \
results. \n <</SYS>> \n\n [INST] Generate THREE Google search queries that \
are similar to this question. The output should be a numbered list of questions \
and each should have a question mark at the end: \n\n {question} [/INST]""",
)

DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an assistant tasked with improving Google search \
results. Generate THREE Google search queries that are similar to \
this question. The output should be a numbered list of questions and each \
should have a question mark at the end: {question}""",
)


class LineList(BaseModel):
    """List of questions."""

    lines: List[str] = Field(description="Questions")


class QuestionListOutputParser(PydanticOutputParser):
    """Output parser for a list of numbered questions."""

    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = re.findall(r"\d+\..*?(?:\n|$)", text)
        return LineList(lines=lines)


class WebResearchRetriever(BaseRetriever):
    """`Google Search API` retriever."""

    # Inputs
    vectorstore: VectorStore = Field(
        ..., description="Vector store for storing web pages"
    )
    llm_chain: LLMChain
    search: GoogleSearchAPIWrapper = Field(..., description="Google Search API Wrapper")
    num_search_results: int = Field(1, description="Number of pages per Google search")
    text_splitter: TextSplitter = Field(
        RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50),
        description="Text splitter for splitting web pages into chunks",
    )
    url_database: List[str] = Field(
        default_factory=list, description="List of processed URLs"
    )
    verify_ssl: bool = Field(
        True,
        description="Whether to verify SSL certificates.",
    )

    @classmethod
    def from_llm(
        cls,
        vectorstore: VectorStore,
        llm: BaseLLM,
        search: GoogleSearchAPIWrapper,
        prompt: Optional[BasePromptTemplate] = None,
        num_search_results: int = 1,
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=150
        ),
        verify_ssl: bool = True,
    ) -> "WebResearchRetriever":
        """Initialize from llm using default template.

        Args:
            vectorstore: Vector store for storing web pages
            llm: llm for search question generation
            search: GoogleSearchAPIWrapper
            prompt: prompt to generating search questions
            num_search_results: Number of pages per Google search
            text_splitter: Text splitter for splitting web pages into chunks
            verify_ssl: Indicates whether to verify SSL certificates

        Returns:
            WebResearchRetriever
        """

        if not prompt:
            QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
                default_prompt=DEFAULT_SEARCH_PROMPT,
                conditionals=[
                    (lambda llm: isinstance(llm, LlamaCpp), DEFAULT_LLAMA_SEARCH_PROMPT)
                ],
            )
            prompt = QUESTION_PROMPT_SELECTOR.get_prompt(llm)

        # Use chat model prompt
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            output_parser=QuestionListOutputParser(),
        )

        return cls(
            vectorstore=vectorstore,
            llm_chain=llm_chain,
            search=search,
            num_search_results=num_search_results,
            text_splitter=text_splitter,
            verify_ssl=verify_ssl,
        )

    def clean_search_query(self, query: str) -> str:
        # Some search tools (e.g., Google) will
        # fail to return results if query has a
        # leading digit: 1. "LangCh..."
        # Check if the first character is a digit
        if query[0].isdigit():
            # Find the position of the first quote
            first_quote_pos = query.find('"')
            if first_quote_pos != -1:
                # Extract the part of the string after the quote
                query = query[first_quote_pos + 1 :]
                # Remove the trailing quote if present
                if query.endswith('"'):
                    query = query[:-1]
        return query.strip()

    def search_tool(self, query: str, num_search_results: int = 1) -> List[dict]:
        """Returns num_search_results pages per Google search."""
        query_clean = self.clean_search_query(query)
        result = self.search.results(query_clean, num_search_results)
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
        logger.info("Searching for relevant urls...")
        urls_to_look = []
        for query in questions:
            # Google search
            search_results = self.search_tool(query, self.num_search_results)
            logger.info("Searching for relevant urls...")
            logger.info(f"Search results: {search_results}")
            for res in search_results:
                if res.get("link", None):
                    urls_to_look.append(res["link"])

        # Relevant urls
        urls = set(urls_to_look)

        # Check for any new urls that we have not processed
        new_urls = list(urls.difference(self.url_database))

        logger.info(f"New URLs to load: {new_urls}")

        # Load, split, and add new urls to vectorstore
        if new_urls:

            # Split urls by type
            html_urls, pdf_urls = self._split_url_by_type(new_urls)

            docs = []
            if len(html_urls) > 0:
                html_loader = AsyncHtmlLoader(html_urls, verify_ssl=self.verify_ssl)
                html2text = Html2TextTransformer()
                logger.info("Indexing new urls...")
                docs = html_loader.load()
                docs = list(html2text.transform_documents(docs))

            if len(pdf_urls) > 0:
                pdf_loader = AsyncPdfLoader(pdf_urls, retries=1, verify_ssl=self.verify_ssl)
                docs = docs + pdf_loader.load()

            # add_documents will throw an error if the doc list is empty, which can happen if we were unable
            # to decode any of the URL targets.
            if docs is not None and len(docs) > 0:
                docs = self.text_splitter.split_documents(docs)
                self.vectorstore.add_documents(docs)

            self.url_database.extend(new_urls)

        # Search for relevant splits
        # TODO: make this async
        logger.info("Grabbing most relevant splits from urls...")
        docs = []
        for query in questions:
            clean_query = self.clean_search_query(query)
            docs.extend(self.vectorstore.similarity_search(clean_query))

        # Get unique docs
        unique_documents_dict = {
            (doc.page_content, tuple(sorted(doc.metadata.items()))): doc for doc in docs
        }
        unique_documents = list(unique_documents_dict.values())
        return unique_documents

    def _split_url_by_type(self, urls: List[str]) -> (List[str], List[str]):
        """
        Split urls by type (html, pdf).
        :param urls: the list of urls to split.
        :return: tuple of (html_urls, pdf_urls).
        """

        # Prepare context to disable SSL verification if needed
        if self.verify_ssl:
            ctx = None
        else:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

        # Get info on each URL and assign to the appropriate list
        headers = get_request_headers()
        html_urls = []
        pdf_urls = []
        for url in urls:
            try:
                req = Request(url, headers=headers)
                con = urlopen(req, timeout=60, context=ctx)
                doc_type = con.info()['content-type']

                if 'text/html' in doc_type:
                    html_urls.append(url)
                elif 'application/pdf' in doc_type:
                    pdf_urls.append(url)
            except Exception as e:
                logger.warning(f"Error while checking url type for '{url}':\n{e}")

                # We couldn't determine the type, so attempt to choose based on the file extension
                if ".pdf" in url:
                    pdf_urls.append(url)
                else:
                    html_urls.append(url)

        return html_urls, pdf_urls

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        raise NotImplementedError