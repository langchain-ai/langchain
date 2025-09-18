from langchain.agents.types import AgentMiddleware, AgentState, ModelRequest, AgentUpdate, AgentJump
from typing_extensions import TypedDict

PROMPT = """Check if the conversation trips any of the guardrails. If it trips multiple, flag the guardrail that is violated the most

<conversation>
{conversation}
</conversation>

<guardrails>
{guardrails}
</guardrails>"""

class Guardrail(TypedDict):
    name: str
    prompt: str
    response_str: str

class InputGuardrailMiddleware(AgentMiddleware):

    def __init__(self, guardrails: list[Guardrail], model):
        super().__init__()
        self.guardrails = guardrails
        self.model = model

    def _convert_to_string(self, state: AgentState):
        # TODO: improve
        return str(state['messages'])

    def before_model(self, state: AgentState) -> AgentUpdate | AgentJump | None:
        conversation = self._convert_to_string(state)
        guardrails = "\n".join([
            f"<{guard['name']}>{guard['prompt']}</{guard['name']}>" for guard in self.guardrails
        ])
        prompt = PROMPT.format(conversation=conversation, guardrails=guardrails)

        class Response(TypedDict):
            # todo: fix docstring
            """flagged should be one of {} or `none`"""
            flagged: str

        response = self.model.with_structured_output(Response).invoke(prompt)
        if response['flagged'] == 'none':
            return
        else:
            resp = {g['name']: g['response_str'] for g in self.guardrails}
            return {
                "messages": [{"role": 'ai', "content": resp}],
                "jump_to": "__end__"
            }



