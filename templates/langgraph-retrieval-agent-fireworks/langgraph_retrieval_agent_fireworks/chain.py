import operator
import sys
from typing import Annotated, List, TypedDict, Union

from langchain import hub
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.tools.render import render_text_description
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_fireworks.chat_models import ChatFireworks
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor

MODEL_ID = "accounts/fireworks/models/mixtral-8x7b-instruct"


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


# Set up tool(s)
description = (
    "A wrapper around Arxiv.org "
    "Useful for when you need to answer questions about Physics, Mathematics, "
    "Computer Science, Quantitative Biology, Quantitative Finance, Statistics, "
    "Electrical Engineering, and Economics "
    "from scientific articles on arxiv.org. "
    "Input should be a search query."
)
arxiv_tool = create_retriever_tool(ArxivRetriever(), "arxiv", description)
tools = [arxiv_tool]

# Set up LLM
llm = ChatFireworks(
    model=MODEL_ID,
    temperature=0,
    max_tokens=2048,
)

# Setup ReAct style prompt
prompt = hub.pull("tjaffri/react-json")

# This prompt requires:
# - tools (names and descriptions)
# - tool_names (names only)
# - input (question)
# - agent_scratchpad

# We can fill out the first two using the tools array
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

# Define the agent using LangGraph
# Ref: https://github.com/langchain-ai/langgraph/blob/main/examples/agent_executor/base.ipynb

llm_with_stop = llm.bind(stop=["\nObservation"])
agent_runnable = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
    }
    | prompt
    | llm_with_stop
    | ReActJsonSingleInputOutputParser()
)


class AgentState(TypedDict):
    # The input string
    input: str
    # The list of previous messages in the conversation
    chat_history: list[BaseMessage]
    # The outcome of a given call to the agent
    # Needs `None` as a valid type, since this is what this will start as
    agent_outcome: Union[AgentAction, AgentFinish, None]
    # List of actions and corresponding observations
    # Here we annotate this with `operator.add` to indicate that operations to
    # this state should be ADDED to the existing values (not overwrite it)
    agent_scratchpad: Annotated[list[tuple[AgentAction, str]], operator.add]


# This a helper class we have that is useful for running tools
# It takes in an agent action and calls that tool and returns the result
tool_executor = ToolExecutor(tools)


# Define the agent
def run_agent(data, config):
    agent_outcome = agent_runnable.invoke(data, config)
    return {"agent_outcome": agent_outcome}


# Define the function to execute tools
def execute_tools(data, config):
    # Get the most recent agent_outcome - this is the key added in the `agent` above
    agent_action = data["agent_outcome"]
    output = tool_executor.invoke(agent_action, config)
    return {"agent_scratchpad": [(agent_action, str(output))]}


# Define logic that will be used to determine which conditional edge to go down
def should_continue(data):
    # If the agent outcome is an AgentFinish, then we return `exit` string
    # This will be used when setting up the graph to define the flow
    if isinstance(data["agent_outcome"], AgentFinish):
        return "end"
    # Otherwise, an AgentAction is returned
    # Here we return `continue` string
    # This will be used when setting up the graph to define the flow
    else:
        return "continue"


# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()


class InputType(BaseModel):
    input: str = ""


# AgentExecuture is the langgraph runnable with the given input type
agent_executor = app.with_types(input_type=InputType)

if __name__ == "__main__":
    if sys.gettrace():
        # This code will only run if a debugger is attached
        output = agent_executor.invoke(
            {"input": "When was the attention is all you need paper published?"}
        )
        print(output)
