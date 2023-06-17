"""Test RetrievalQA functionality."""
from pathlib import Path
from typing import Any, List, Optional

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.tracers.base import BaseTracer
from langchain.callbacks.tracers.schemas import Run, RunTypeEnum
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.context_builder import ContextBuilder
from langchain.chains.loading import load_chain
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import LLMResult
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


class SourceContextBuilder(ContextBuilder):
    def create_context(
        self, docs: List[Document], document_prompt: BasePromptTemplate
    ) -> str:
        return ",".join([doc.metadata["source"] for doc in docs])


class TestCallbackHandler(BaseTracer):
    name = "test_callback_handler"
    prompt_after_formatting: str

    def _persist_run(self, run: Run) -> None:
        pass

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        self.prompt_after_formatting = text


def test_retrieval_qa_saving_loading(tmp_path: Path) -> None:
    """Test saving and loading."""
    loader = TextLoader("docs/extras/modules/state_of_the_union.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type_kwargs={"context_builder": SourceContextBuilder()},
        retriever=docsearch.as_retriever(),
    )

    qa.combine_documents_chain.llm_chain.verbose = True
    handler = TestCallbackHandler()
    qa("What is the capital of Ireland?", callbacks=[handler])
    assert """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

docs/extras/modules/state_of_the_union.txt,docs/extras/modules/state_of_the_union.txt,docs/extras/modules/state_of_the_union.txt,docs/extras/modules/state_of_the_union.txt

Question: What is the capital of Ireland?""".replace(
        "\n", ""
    ) in handler.prompt_after_formatting.strip().replace(
        "\n", ""
    )
