from langchain.agents.reflexion.base import BaseReflector

class EmptyReflector(BaseReflector):
    """Use this class if you don't want reflexion."""

    def should_reflect(self) -> bool:
        """Never reflect"""
        return False

    def get_history(self, current_trial_number: int) -> str:
        """No reflection history"""
        return ""
