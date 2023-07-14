from typing import Any, List, Optional, Tuple

from langchain.agents.agent import Reflector, ReflexionOutputParser
from langchain.agents.reflexion.alfworld_prompt import ALFWORLD_PROMPT
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import AgentAction


class ReactReflexionOutputParser(ReflexionOutputParser):
    def parse(self, text: str) -> str:
        # The Reflexion prompt asks the LLM to complete after "New plan: ",
        # so the entire result is the reflexion
        return text

    @property
    def _type(self) -> str:
        return "react_reflexion_output_parser"


class ReactReflector(Reflector):
    max_action_repetition: Optional[int] = 2
    output_parser: ReflexionOutputParser = ReactReflexionOutputParser()

    def get_history(self, current_trial_number: int) -> str:
        if current_trial_number == 1:
            # In 1st trial, we have not done any reflection yet
            return ""

        history = ""
        # # We create a trial history of form:
        #
        # Trial 1:
        # Thought: ... \nAction: ... \nObservation: ...
        # STATUS: FAIL
        # New plan: I should have ...
        #
        # Trial 2: ...
        iter = enumerate(zip(self.trial_history, self.trial_reflexions))
        for trial_no, (trial, reflexion) in iter:
            history += "\n" + self.current_trial_prefix(trial_no)
            history += "\n" + trial
            history += "\n" + self.trial_suffix
            history += reflexion + "\n"

        # Lastly, we add the prefix of the current trial (e.g. "Trial 3: \n")
        history += self.current_trial_prefix(current_trial_number)

        return history

    @classmethod
    def create_prompt(self) -> BasePromptTemplate:
        return ALFWORLD_PROMPT

    def should_reflect(
        self,
        iterations_in_trial: int,
        execution_time_in_trial: float,
        *args: Any,
        **kwargs: Any
    ) -> bool:
        # We reflect when ...
        # we have too many iterations in trial or trial took too long, or
        if super().should_reflect(iterations_in_trial, execution_time_in_trial):
            return True
        # ... or we're stuck in an action loop, or
        if "intermediate_steps" not in kwargs:
            raise ValueError(
                "ReactReflector.should_reflect expects" " intermediate_steps as kwarg"
            )
        intermediate_steps = kwargs["intermediate_steps"]
        if (
            self.max_action_repetition is not None
            and self._count_repetitions(intermediate_steps)
            >= self.max_action_repetition
        ):
            print("Trial failed due to _max_action_repetition reached")
            return True
        # ... we're done, but the task was not succesful
        # TODO for a future version:
        # When agents finishes, check if outputs answers initia question
        # (via LLM call or string comparison [e.g. contains "don't know"])

        return False

    @staticmethod
    def _count_repetitions(intermediate_steps: List[Tuple[AgentAction, str]]) -> int:
        last_tool = intermediate_steps[-1][0].tool
        last_tool_input = intermediate_steps[-1][0].tool_input

        count = 0

        # Iterate over intermediate_steps in reverse order,
        # starting from second last element
        for action, _ in reversed(intermediate_steps[:-1]):
            if action.tool == last_tool and action.tool_input == last_tool_input:
                count += 1
            else:
                break

        if count == 0:
            return 0
        else:
            # 1 repitition means the same action was found twice
            return count + 1

    def reflect(
        self,
        input: str,
        current_trial: str,
        current_trial_no: int,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> str:
        """
        Given agent history, decide what agent can do better next time.

        Args:
            input: user input
            current_trial: string representation of current trial
            current_trial_no: number of current trial

        Returns:
            Reflection obtained from LLM.
        """

        scratchpad = (
            self.get_history(current_trial_no) + current_trial + self.trial_suffix
        )
        inputs = {"input": input, "agent_scratchpad": scratchpad}
        output = self.llm_chain.predict(
            callbacks=run_manager.get_child() if run_manager else None, **inputs
        )
        reflexion = self.output_parser.parse(output)

        if run_manager:
            run_manager.on_reflection(text="Reflection: " + reflexion, color="green")

        # Save reflexion history
        self.trial_history.append(current_trial)
        self.trial_reflexions.append(reflexion)

        return reflexion

    async def areflect(
        self,
        input: str,
        current_trial: str,
        current_trial_no: int,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> str:
        """
        Given agent history, decide what agent can do better next time.

        Args:
            input: user input
            current_trial: string representation of current trial
            current_trial_no: number of current trial

        Returns:
            Reflection obtained from LLM.
        """

        scratchpad = (
            self.get_history(current_trial_no) + current_trial + self.trial_suffix
        )
        inputs = {"input": input, "agent_scratchpad": scratchpad}
        output = self.llm_chain.predict(
            callbacks=run_manager.get_child() if run_manager else None, **inputs
        )
        reflexion = self.output_parser.parse(output)

        if run_manager:
            text = "Reflection: " + reflexion
            await run_manager.on_reflection(text=text, color="green")

        # Save reflexion history
        self.trial_history.append(current_trial)
        self.trial_reflexions.append(reflexion)

        return reflexion

    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> ReflexionOutputParser:
        return ReactReflexionOutputParser()
