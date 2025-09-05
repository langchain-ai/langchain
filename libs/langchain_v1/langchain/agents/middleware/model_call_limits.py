import operator
from typing import Annotated

from langchain.agents.types import AgentJump, AgentMiddleware, AgentState, AgentUpdate


class State(AgentState):
    model_request_count: Annotated[int, operator.add]


class ModelRequestLimitMiddleware(AgentMiddleware):
    """Terminates after N model requests"""

    state_schema = State

    def __init__(self, max_requests: int = 10):
        self.max_requests = max_requests

    def before_model(self, state: State) -> AgentUpdate | AgentJump | None:
        # TODO: want to be able to configure end behavior here
        if state.get("model_request_count", 0) == self.max_requests:
            return {"jump_to": "__end__"}

        return {"model_request_count": 1}
