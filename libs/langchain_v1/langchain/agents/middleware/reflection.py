from langchain.agents.types import AgentMiddleware, AgentState, ModelRequest, AgentUpdate, AgentJump

class ReflectionMiddleware(AgentMiddleware):

    def __init__(self, reflection_step):
        self.reflection_step = reflection_step

    def after_model(self, state: AgentState) -> AgentUpdate | AgentJump | None:
        reflection = self.reflection_step(state)
        if reflection:
            return {
                "messages": [{'role': 'user', 'content': reflection}],
                "jump_to": "model"
            }
