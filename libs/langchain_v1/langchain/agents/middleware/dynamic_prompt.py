from langchain.agents.types import AgentMiddleware, AgentState, ModelRequest

class DynamicPrompt(AgentMiddleware):

    def __init__(self, modifier):
        self.modifier = modifier

    def modify_model_request(self, request: ModelRequest, state) -> ModelRequest:
        prompt = self.modifier(state)
        request.system_prompt = prompt
        return request


class DynamicMessages(AgentMiddleware):

    def __init__(self, modifier):
        self.modifier = modifier

    def modify_model_request(self, request: ModelRequest, state) -> ModelRequest:
        messages = self.modifier(state)
        request.messages = messages
        return request
