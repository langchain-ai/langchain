import os
import pickle
from typing import Dict, List, Tuple

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.vectorstores.pinecone import Pinecone
from langchain.storage.in_memory import InMemoryStore
from langchain.tools.base import BaseTool
from langchain.prompts import MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.runnable import Runnable, RunnableLambda, RunnableParallel
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

if os.environ.get("OPENAI_API_KEY", None) is None:
    raise Exception("Missing `OPENAI_API_KEY` environment variable.")

# Pinecone options (please see README for notes on how to run indexing)
if os.environ.get("PINECONE_API_KEY", None) is None:
    raise Exception("Missing `PINECONE_API_KEY` environment variable.")

if os.environ.get("PINECONE_ENVIRONMENT", None) is None:
    raise Exception("Missing `PINECONE_ENVIRONMENT` environment variable.")

DOCUGAMI_DOCSET_ID = "fi6vi49cmeac"
DOCUGAMI_DOCSET_NAME = "Earnings Calls"
DOCUGAMI_DOCSET_DESCRIPTION = "This type of document is an edited transcript of a corporate earnings conference call, providing information about a company's financial performance, leadership team, and forward-looking statements about earnings."

PINECONE_INDEX_NAME = (
    os.environ.get("PINECONE_INDEX", "langchain-docugami") + f"-{DOCUGAMI_DOCSET_ID}"
)

PARENT_DOC_STORE_PATH = os.environ.get(
    "PARENT_DOC_STORE_ROOT_PATH", "temp/parent_docs.pkl"
)

# LangSmith options (set for tracing)
LANGCHAIN_TRACING_V2 = True
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "")

llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
embeddings = OpenAIEmbeddings()

# Chunks are in the vector store, and full documents are in an inmemory store
chunk_vectorstore = Pinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings)
with open(PARENT_DOC_STORE_PATH, "rb") as file:
    parent_docstore: InMemoryStore = pickle.load(file)

retriever = MultiVectorRetriever(
    vectorstore=chunk_vectorstore,
    docstore=parent_docstore,
    search_kwargs={"k": 10},  # retrieve more small chunks from the vector store
    docstore_k=2,  # provide fewer large chunks from the docstore
)

docset_retrieval_tool = create_retriever_tool(
    retriever,
    "search_earnings_calls",
    f"Searches and returns documents from the {DOCUGAMI_DOCSET_NAME} document set. {DOCUGAMI_DOCSET_DESCRIPTION}.",
)
tools: List[BaseTool] = [docset_retrieval_tool]


assistant_system_message = """You are a helpful assistant. \
Use tools (only if necessary) to best answer the users questions."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", assistant_system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


def llm_with_tools(input: Dict) -> Runnable:
    return RunnableLambda(lambda x: x["input"]) | llm.bind(functions=input["functions"])


def _format_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


agent = (
    RunnableParallel(
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: _format_chat_history(x["chat_history"]),
            "agent_scratchpad": lambda x: format_to_openai_functions(
                x["intermediate_steps"]
            ),
            "functions": lambda x: [
                format_tool_to_openai_function(tool) for tool in tools
            ],
        }
    )
    | {
        "input": prompt,
        "functions": lambda x: x["functions"],
    }
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)


class AgentInput(BaseModel):
    input: str = ""
    chat_history: List[Tuple[str, str]] = Field(
        ..., extra={"widget": {"type": "chat", "input": "input", "output": "output"}}
    )


chain = AgentExecutor(agent=agent, tools=tools).with_types(input_type=AgentInput)
