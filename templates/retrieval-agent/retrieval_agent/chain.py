import os
from typing import List, Tuple

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.retrievers import BaseRetriever
from langchain_openai import AzureChatOpenAI


class ArxivRetriever(BaseRetriever, ArxivAPIWrapper):
    """`Arxiv` retriever.

    It wraps load() to get_relevant_documents().
    It uses all ArxivAPIWrapper arguments without any change.
    """

    get_full_documents: bool = False

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        try:
            if self.is_arxiv_identifier(query):
                results = self.arxiv_search(
                    id_list=query.split(),
                    max_results=self.top_k_results,
                ).results()
            else:
                results = self.arxiv_search(  # type: ignore
                    query[: self.ARXIV_MAX_QUERY_LENGTH], max_results=self.top_k_results
                ).results()
        except self.arxiv_exceptions as ex:
            return [Document(page_content=f"Arxiv exception: {ex}")]
        docs = [
            Document(
                page_content=result.summary,
                metadata={
                    "Published": result.updated.date(),
                    "Title": result.title,
                    "Authors": ", ".join(a.name for a in result.authors),
                },
            )
            for result in results
        ]
        return docs


description = (
    "A wrapper around Arxiv.org "
    "Useful for when you need to answer questions about Physics, Mathematics, "
    "Computer Science, Quantitative Biology, Quantitative Finance, Statistics, "
    "Electrical Engineering, and Economics "
    "from scientific articles on arxiv.org. "
    "Input should be a search query."
)

# Create the tool
arxiv_tool = create_retriever_tool(ArxivRetriever(), "arxiv", description)
tools = [arxiv_tool]
llm = AzureChatOpenAI(
    temperature=0,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
assistant_system_message = """You are a helpful research assistant. \
Lookup relevant information as needed."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", assistant_system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])


def _format_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


agent = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)


class AgentInput(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(
        ..., extra={"widget": {"type": "chat", "input": "input", "output": "output"}}
    )


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True).with_types(
    input_type=AgentInput
)
