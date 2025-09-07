import uuid

from langchain.agents.types import AgentMiddleware, AgentState, ModelRequest, AgentUpdate, AgentJump
from typing_extensions import TypedDict, Type
from langchain.tools import tool
from langchain.chat_models import init_chat_model

from langchain.agents.middleware._utils import _generate_correction_tool_messages


class Agent(TypedDict):
    name: str
    description: str
    prompt: str
    tools: list
    model: str
    model_settings: dict
    response_format: Type


class SwarmAgentState(AgentState):
    active_agent: str | None


class SwarmMiddleware(AgentMiddleware):

    state_schema = SwarmAgentState

    def __init__(self, agents: list[Agent], starting_agent: str):
        self.agents = agents
        self.starting_agent = starting_agent
        self.agent_mapping = {a['name']: a for a in agents}

    @property
    def tools(self):
        return [t for a in self.agents for t in a['tools']]

    def _get_handoff_tool(self, agent: Agent):
        @tool(
            name_or_callable=f"handoff_to_{agent['name']}",
            description=f"Handoff to agent {agent['name']}. Description of this agent:\n\n{agent['description']}"
        )
        def handoff():
            pass

        return handoff


    def _get_main_handoff_tools(self):
        tools = []
        for agent in self.agents:
            tools.append(self._get_handoff_tool(agent))
        return tools


    def modify_model_request(self, request: ModelRequest, state: SwarmAgentState) -> ModelRequest:
        if state.get('active_agent') is None:
            request.tools = request.tools + self._get_main_handoff_tools()
            return request
        active_agent = self.agent_mapping[state['active_agent']]
        request.system_prompt = active_agent['prompt']
        request.tools = active_agent['tools']
        if 'model' in active_agent:
            request.model = init_chat_model(active_agent['model'])
        if 'model_settings' in active_agent:
            request.model_settings = active_agent['model_settings']
        if 'response_format' in active_agent:
            request.response_format = active_agent['response_format']
        return request

    def after_model(self, state: SwarmAgentState) -> AgentUpdate | AgentJump | None:
        messages = state["messages"]
        active_agent = state.get('active_agent')
        if not messages:
            return None

        last_message = messages[-1]

        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            if active_agent is None:
                return None
            else:
                fake_tool_call_id = str(uuid.uuid4())
                last_message.tool_calls = [{
                "type": "tool_call",
                "name": "hand_back_to_supervisor",
                "args": {},
                "id": fake_tool_call_id,
            }]
                fake_tool_message = {"role": "tool", "content": "Handed back to supervisor", "tool_call_id": fake_tool_call_id}
                return {
                    "messages": [last_message, fake_tool_message],
                    "jump_to": "model"
                }
        if active_agent is not None:
            return None
        handoff_tools = self._get_main_handoff_tools()
        handoff_tool_names = [t.name for t in handoff_tools]

        handoffs = []
        for tool_call in last_message.tool_calls:
            if tool_call['name'] in handoff_tool_names:
                handoffs.append(tool_call)
        if len(handoffs) == 0:
            return
        elif len(handoffs) > 1:
            msg = "Multiple handoffs at the same time are not supported, please just call one at a time."
            return {
                "messages": _generate_correction_tool_messages(msg,
                                                               last_message.tool_calls),
                "jump_to": "model"
            }
        else:
            tool_call = handoffs[0]
            handoff_to = tool_call['name'][len("handoff_to_"):]
            return {
                "messages":[{
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": f"Handed off to agent {handoff_to}",
            }],
                "active_agent":handoff_to,
                "jump_to": "model"
            }





