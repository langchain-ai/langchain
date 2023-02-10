"""Rich structured data about each iteration of the agent loop."""

from typing import Any, Optional, Tuple, Union

from pydantic import BaseModel

from langchain.schema import AgentAction, AgentFinish


class StepOutput(BaseModel):
    """The output produced by a single step of the agent loop.

    You can subclass this to store rich structured data instead of relying solely on
    AgentAction's str inputs.
    """

    decision: Union[AgentAction, AgentFinish]
    """The decision taken during the step.

    This could be deciding to use a tool, or deciding to end the loop.
    """
    observation: Optional[str]
    """The observed effects of the previous action.

    Only present if the decision taken wasn't to end the loop.
    """

    @property
    def is_finish(self) -> bool:
        """Whether this step signals the end of the agent loop."""
        return isinstance(self.decision, AgentFinish)

    def log(self, **kwargs: Any) -> str:
        """Produce a log for this step.

        This is usually useful for constructing the scratchpad. The kwargs are there to
        allow for flexibility during scratchpad construction. Subclass `StepOutput` and
        override this function to make use of them.
        """
        if len(kwargs) > 0:
            raise ValueError("The default log doesn't take any options")
        assert self.observation is not None
        # this is just an example because this function isn't actually used in any of
        # the current Agent logic, but if kwargs were to contain `observation_prefix`
        # and `llm_prefix`, then we could recreate the scratchpad logic in
        # `Agent._construct_scratchpad`
        return self.decision.log + "\n" + self.observation

    def as_intermediate_step(self) -> Tuple[AgentAction, str]:
        """Compatibility function for existing action-observation tuple."""
        assert isinstance(self.decision, AgentAction)
        assert self.observation is not None  # should follow from first assert
        return (self.decision, self.observation)
