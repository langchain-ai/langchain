import logging
import re
from typing import Any, List, Optional

from langchain.chains import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from pydantic import BaseModel, Field

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.llms import LlamaCpp
from langchain_community.utilities import GoogleSearchAPIWrapper

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


class QuestionListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of numbered questions."""

    def parse(self, text: str) -> List[str]:
        lines = re.findall(r"\d+\..*?(?:\n|$)", text)
        return lines


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
    trust_env: bool = Field(
        False,
        description="Whether to use the http_proxy/https_proxy env variables or "
        "check .netrc for proxy configuration",
    )

    allow_dangerous_requests: bool = False
    """A flag to force users to acknowledge the risks of SSRF attacks when using 
    this retriever.
    
    Users should set this flag to `True` if they have taken the necessary precautions
    to prevent SSRF attacks when using this retriever.
    
    For example, users can run the requests through a properly configured
    proxy and prevent the crawler from accidentally crawling internal resources.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the retriever."""
        allow_dangerous_requests = kwargs.get("allow_dangerous_requests", False)
        if not allow_dangerous_requests:
            raise ValueError(
                "WebResearchRetriever crawls URLs surfaced through "
                "the provided search engine. It is possible that some of those URLs "
                "will end up pointing to machines residing on an internal network, "
                "leading"
                "to an SSRF (Server-Side Request Forgery) attack. "
                "To protect yourself against that risk, you can run the requests "
                "through a proxy and prevent the crawler from accidentally crawling "
                "internal resources."
                "If've taken the necessary precautions, you can set "
                "`allow_dangerous_requests` to `True`."
            )
        super().__init__(**kwargs)

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
        trust_env: bool = False,
        allow_dangerous_requests: bool = False,
    ) -> "WebResearchRetriever":
        """Initialize from llm using default template.

        Args:
            vectorstore: Vector store for storing web pages
            llm: llm for search question generation
            search: GoogleSearchAPIWrapper
            prompt: prompt to generating search questions
            num_search_results: Number of pages per Google search
            text_splitter: Text splitter for splitting web pages into chunks
            trust_env: Whether to use the http_proxy/https_proxy env variables
                or check .netrc for proxy configuration
            allow_dangerous_requests: A flag to force users to acknowledge
                the risks of SSRF attacks when using this retriever

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
            trust_env=trust_env,
            allow_dangerous_requests=allow_dangerous_requests,
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
        questions = result["text"]
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
            loader = AsyncHtmlLoader(
                new_urls, ignore_load_errors=True, trust_env=self.trust_env
            )
            html2text = Html2TextTransformer()
            logger.info("Indexing new urls...")
            docs = loader.load()
            docs = list(html2text.transform_documents(docs))
            docs = self.text_splitter.split_documents(docs)
            self.vectorstore.add_documents(docs)
            self.url_database.extend(new_urls)

        # Search for relevant splits
        # TODO: make this async
        logger.info("Grabbing most relevant splits from urls...")
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
