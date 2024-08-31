from __future__ import annotations

from typing import Any, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.tools.playwright.base import BaseBrowserTool
from langchain_community.tools.playwright.utils import (
    aget_current_page,
    get_current_page,
)
from langchain_community.vectorstores import FAISS


class RetrievalExtractTextTool(BaseBrowserTool):
    """Tool for extracting and retrieve relevant text on the current webpage."""

    name: str = "retrieval_extract_text"
    description: str = "Extract relevant text on the current webpage"
    args_schema: Type[BaseModel] = BaseModel
    prompt: ChatPromptTemplate

    def __init__(self, /, **kwargs: Any) -> None:
        try:
            from bs4 import BeautifulSoup  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'beautifulsoup4' package is required to use this tool."
                " Please install it with 'pip install beautifulsoup4'."
            )
        super().__init__(**kwargs)

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        if self._validate_prompt():
            retriever_message: HumanMessage | None = next(
                (msg for msg in self.prompt.messages if isinstance(msg, HumanMessage)),
                None,
            )

            # Use Beautiful Soup since it's faster than looping through the elements
            from bs4 import BeautifulSoup

            if self.sync_browser is None:
                raise ValueError(f"Synchronous browser not provided to {self.name}")

            page = get_current_page(self.sync_browser)
            html_content = page.content()

            # Parse the HTML content with BeautifulSoup
            soup = BeautifulSoup(html_content, "lxml")

            if retriever_message is None:
                return " ".join(text for text in soup.stripped_strings)
            else:
                # Retrieve relevant text
                retriever_query = retriever_message.content
                full_text = "".join(text for text in soup.stripped_strings)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=250, chunk_overlap=12
                )
                chunk_list = text_splitter.create_documents(texts=[full_text])
                embeddings = OpenAIEmbeddings()
                db = FAISS.from_documents(chunk_list, embeddings)
                retriever = db.as_retriever()
                docs = retriever.invoke(str(retriever_query))
                return " ".join(doc.page_content for doc in docs)
        else:
            raise ValueError(
                "RetrievalExtractTextTool does not have attribute 'prompt'. "
                "Please use the injector in "
                "'langchain_core.prompts.injector' "
                "to inject prompt to the tool. "
            )

    async def _arun(
        self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        if self._validate_prompt():
            retriever_message: HumanMessage | None = next(
                (msg for msg in self.prompt.messages if isinstance(msg, HumanMessage)),
                None,
            )

            # Use Beautiful Soup since it's faster than looping through the elements
            from bs4 import BeautifulSoup

            if self.async_browser is None:
                raise ValueError(f"Synchronous browser not provided to {self.name}")

            page = await aget_current_page(self.async_browser)
            html_content = await page.content()

            # Parse the HTML content with BeautifulSoup
            soup = BeautifulSoup(html_content, "lxml")

            if retriever_message is None:
                return " ".join(text for text in soup.stripped_strings)
            else:
                # Retrieve relevant text
                retriever_query = retriever_message.content
                full_text = "".join(text for text in soup.stripped_strings)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=250, chunk_overlap=12
                )
                chunk_list = text_splitter.create_documents(texts=[full_text])
                embeddings = OpenAIEmbeddings()
                db = FAISS.from_documents(chunk_list, embeddings)
                retriever = db.as_retriever()
                docs = retriever.invoke(str(retriever_query))
                return " ".join(doc.page_content for doc in docs)
        else:
            raise ValueError(
                "RetrievalExtractTextTool does not have attribute 'prompt'. "
                ""
                "Please use the injector in "
                "'langchain_core.prompts.injector' "
                "to inject prompt to the tool. "
            )

    def _validate_prompt(self) -> bool:
        if hasattr(self, "prompt"):
            if hasattr(self.prompt, "messages"):
                return True
        return False
