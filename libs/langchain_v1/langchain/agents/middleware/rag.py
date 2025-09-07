from langchain.agents.types import AgentMiddleware, AgentState, ModelRequest, AgentJump, AgentUpdate
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
import uuid

class RAGMiddleware(AgentMiddleware):

    @classmethod
    def from_retriever(cls, retriever: BaseRetriever, description: str):
        @tool(description=description)
        def retrieve(query: str):
            return retriever.get_relevant_documents(query)

        return cls(retrieve)


    def __init__(self, tool):
        self.tool = tool

    @property
    def tools(self):
        return [self.tool]

    def before_model(self, state: AgentState) -> AgentUpdate | AgentJump | None:
        if len(state['messages']) == 1:
            forced_tool_call = {
                "type": "tool_call",
                "name": self.tool.name,
                "args": {"query": state['messages'][0].content},
                "id": str(uuid.uuid4()),
            }
            return {
                "messages": [{"role": "ai", "content": None, "tool_calls": [forced_tool_call]}],
                "jump_to": "tools"
            }
