
from typing import Any, Dict, List

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from pydantic import model_validator


import httpx
from enum import Enum
import os
import time
from langchain_core.vectorstores import VectorStore


class Status(Enum):
    PENDING = 'Pending'
    INPROGRESS = 'InProgress'
    COMPLETED = 'Completed'
    FAILED = 'Failed'


class AnsweritSDK:
    def __init__(self, answerit_url: str, authorization_token: str):
        self.answerit_url = answerit_url if answerit_url else os.environ.get("NIMBLE_ANSWERIT_URL")
        if not self.answerit_url:
            raise WorkflowAuth("Missing Nimble Answer It API endpoint")

        self.auth_token = authorization_token if authorization_token else os.environ.get("NIMBLE_API_KEY")
        if not self.auth_token:
            raise WorkflowAuth("Missing Nimble API Key")

        self.client = httpx.Client(
            base_url=self.answerit_url,
            headers={'Authorization': self.auth_token},
            follow_redirects=True
        )

    def create_pipeline(self, request_body: dict) -> dict:
        response = self.client.post(
            url="/pipelines/",
            json=request_body
        )
        response.raise_for_status()
        return response.json()

    def update_pipeline(self, pipeline_id, request_body: dict) -> str:
        response = self.client.patch(
            url=f"/pipelines/{pipeline_id}",
            json=request_body
        )
        response.raise_for_status()
        return response.json()['pipeline_execution_id']

    def get_pipelines(self) -> list[dict]:
        response = self.client.get(f"/pipelines/")
        response.raise_for_status()
        pipelines = response.json()
        return pipelines

    def get_pipeline(self, pipeline_id: str):
        response = self.client.get(f"/pipelines/{pipeline_id}")
        response.raise_for_status()
        pipeline = response.json()
        return pipeline

    def get_pipeline_execution(self, pipeline_id: str, pipeline_execution_id: str):
        response = self.client.get(f"/pipelines/{pipeline_id}/pipeline-executions/{pipeline_execution_id}")
        response.raise_for_status()
        pipeline_execution = response.json()
        return pipeline_execution

    def wait_for_pipeline_execution_to_finish(self, pipeline_id: str, pipeline_execution_id: str):
        pipeline_execution = self.get_pipeline_execution(pipeline_id, pipeline_execution_id)
        attempt_count = 0
        if pipeline_execution['status'] == Status.FAILED.value:
            raise WorkflowFailed()
        while not pipeline_execution['status'] == Status.COMPLETED.value and attempt_count < 30:
            time.sleep(10)
            pipeline_execution = self.get_pipeline_execution(pipeline_id, pipeline_execution_id)
            if pipeline_execution['status'] == Status.FAILED.value:
                raise WorkflowFailed()
            attempt_count += 1
        if attempt_count == 30:
            raise WorkflowTimeout()
        return pipeline_execution


class WorkflowFailed(Exception):
    def __init__(self):
        super().__init__()


class WorkflowTimeout(Exception):
    def __init__(self):
        super().__init__()


class WorkflowAuth(Exception):
    def __init__(self, msg=None):
        super().__init__(msg)


class Pipeline(VectorStore):
    pipeline_id = None
    def __init__(self, answerit_url: str, authorization_token: str, sources: List[Dict] = None):
        self.client = AnsweritSDK(answerit_url, authorization_token)
        self.pipeline_id = self.client.create_pipeline({
            'questions': [], 'sources': sources if sources else []
          })['pipeline_id']

    def add_sources(self, sources: List[Dict], wait=False) -> None:
        pipeline_execution_id = self.client.update_pipeline(
            self.pipeline_id,
            request_body={"sources": sources}
        )
        if wait:
            self.client.wait_for_pipeline_execution_to_finish(
                self.pipeline_id,
                pipeline_execution_id
            )

    def invoke(self, query):
        pipeline_execution_id = self.client.update_pipeline(
            self.pipeline_id,
            request_body={
              'questions': [query],
              'sources': []
            }
        )
        return self.client.wait_for_pipeline_execution_to_finish(
            self.pipeline_id,
            pipeline_execution_id
        )


    def from_texts(cls):
        raise NotImplementedError("Can't use 'from_texts'")

    def similarity_search(self):
        raise NotImplementedError("Can't use 'similarity_search'")
####


class NimbleItRetriever(BaseRetriever):
    """NimbleIt API retriever.

    See detailed instructions here: https://python.langchain.com/v0.2/docs/integrations/retrievers/nimbleit/

    Setup:
        Install ``langchain-nimbleit`` and other dependencies:

        .. code-block:: bash

            pip install -U nimbleit langchain-nimbleit

    AnswerIt init args:
        answerit_url: answerit api endpoint
        authorization_token: nimble authentication token

    Instantiate:
        .. code-block:: python

            retriever = NimbleItRetriever(
                answerit_url,
                authorization_token,
                sources=[
                    {
                        "url": "https://lilianweng.github.io/posts/2023-06-23-agent/",
                        "return_most_relevant_content": False,
                        "depth": 0
                    }
                ]    
            )

            retriever.add_sources(
                    {
                        "url": "https://en.wikipedia.org/wiki/Pizza",
                        "return_most_relevant_content": False,
                        "depth": 0
                    }
                ]
            )
    Usage:
        .. code-block:: python

            query = "What is llm?"

            retriever.invoke(query)

        .. code-block:: none

            [Document(metadata={
                'evidences': [
                    {'evidence': 'ReAct (Yao et al. 2023) integrates reasoning and acting within LLM by extending the action space to be a combination of task-specific discrete actions and the language space.',
                     'url': 'https://lilianweng.github.io/posts/2023-06-23-agent/'},
                    {'evidence': 'The ReAct prompt template incorporates explicit steps for LLM to think, roughly formatted as: Thought: ... Action: ...',
                     'url': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}, 
                    {'evidence': 'Toolformer (Schick et al. 2023) fine-tune a LM to learn to use external tool APIs.', 
                     'url': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}, 
                    {'evidence': 'Plugins and OpenAI API function calling are good examples of LLMs augmented with tool use capability working in practice.',
                     'url': 'https://lilianweng.github.io/posts/2023-06-23-agent/'},
                    {'evidence': 'AutoGPT has drawn a lot of attention into the possibility of setting up autonomous agents with LLM as the main controller.',
                     'url': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}
                ]},
                page_content="LLM stands for 'Large Language Model.' It refers to a type of artificial intelligence model that is designed to understand and generate human language. 
                These models are trained on vast amounts of text data and can perform a variety of language-related tasks, such as translation, summarization, question answering, and more.
                LLMs are capable of generating coherent and contextually relevant text based on the input they receive.
                They are often used in applications like chatbots, virtual assistants, and other AI-driven communication tools.
                Examples of LLMs include OpenAI's GPT-3 and GPT-4, which are known for their advanced language processing capabilities."
            )]
    Use within a chain:
        .. code-block:: python

            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import PromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            from langchain_core.output_parsers import StrOutputParser

            import os
            os.environ["OPENAI_API_KEY"] = "####"

            llm = ChatOpenAI()

            PROMPT_TEMPLATE = \"\"\"
            Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
            Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.

            <context>
            {context}
            </context>

            <question>
            {question}
            </question>

            Assistant:\"\"\"

            prompt = PromptTemplate(
                template=PROMPT_TEMPLATE, input_variables=["context", "question"]
            )

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            retriever = NimbleItRetriever(
                answerit_url="https://answerit.webit.live",
                authorization_token=os.environ["NIMBLE_API_KEY"],
            )

            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            retriever.add_sources([
                {'url': 'https://en.wikipedia.org/wiki/Lila_(Robinson_novel)', 'depth': 0},
                {'url': 'https://www.newyorker.com/magazine/2014/10/06/lonesome-road', 'depth': 0}
            ], wait=True)

            result = rag_chain.invoke("What novels has Lila written and what are their contents?")


        .. code-block:: none

             "Lila, written by Marilynne Robinson, is the titular character in the novel 'Lila'. The novel is the third installment in the Gilead series, ..."

    """

    answerit_url: str = None
    authorization_token: str

    store: Pipeline
    retriever: BaseRetriever

    @model_validator(mode="before")
    @classmethod
    def create_retriever(cls, values: Dict) -> Any:
        """Create the NimbleIt store and retriever."""
        values["store"] = Pipeline(
            values["answerit_url"],
            values["authorization_token"]
        )
        values["retriever"] = values["store"].as_retriever()
        return values

    def add_sources(self, sources: List[dict], wait=False) -> None:
        """Add sources to the NimbleIt Pipeline

        Args:
            sources (List[dict]):
              [
                {"url": URL, "depth": INT, "return_most_relevant_content": BOOLEAN}
              ]
        """
        self.store.add_sources(sources, wait=wait)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        r = self.store.invoke(query)
        return [
            Document(page_content=ans['answer'], metadata={"evidences": ans['evidences']})
            for ans in r['results']['answers']
        ]
