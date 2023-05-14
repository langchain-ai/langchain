from langchain.agents.reflexion.base import BaseReflector

class EmptyReflector(BaseReflector):
    """Agent for the Reflexer chain."""

    def should_reflect(self) -> bool:
        return False
