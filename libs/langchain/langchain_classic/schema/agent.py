from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish

class LangchainClassicAgent:
    def __init__(self):
        self.state = {}

    def update(self, action: AgentAction):
        if action.message_log:
            self.state = {**self.state, **action.message_log}
        if action.finish:
            self.state = action.finish

__all__ = ["AgentAction", "AgentActionMessageLog", "AgentFinish", "LangchainClassicAgent"]