from typing import Any, Dict, List, Optional, Tuple
from langchain.agents.reflexion.alfworld_prompt import ALFWORLD_PROMPT
from langchain.agents.reflexion.base import BaseReflector, ReflexionOutputParser
from langchain.callbacks.manager import Callbacks
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import AgentAction


class ReactReflector(BaseReflector):

    max_action_repetition: Optional[int] = 2

    def __init__(self) -> None:
        self.output_parser = ReactReflexionOutputParser()

    def get_history(self, current_trial_number: int) -> str:
        if current_trial_number==1:
            # In 1st trial, we have not done any reflection yet
            return ""
        
        history = ""
        # # We create a trial histrory of form:
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

    def create_prompt(self) -> BasePromptTemplate:
        return ALFWORLD_PROMPT

    def should_reflect(
            self, 
            iterations_in_trial: int,
            time_elapsed_in_trial: float,
            intermediate_steps: List[Tuple[AgentAction, str]]) -> bool:
        # We reflect when ...
        # we have too many iterations in trial or trial took too long, or
        if super().should_reflect(iterations_in_trial, time_elapsed_in_trial):
            return True
        # ... we're stuck in an action loop, or
        if (self.max_action_repetition is not None
            and self._count_repetitions(intermediate_steps) >= self.max_action_repetition):
            print("Trial failed due to _max_action_repetition reached")
            return True
        # ... we're done, but the task was not succesful
        # TODO

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

        return count

    def reflect(
            self,
            input: str,
            current_trial: str,
            current_trial_no: int,
            callbacks: Callbacks = None) -> str:
        """#TODO
        
        Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """

        scratchpad = (self.get_history(current_trial_no) + current_trial
                      + self.trial_suffix)
        inputs = {
            "input": input,
            "agent_scratchpad": scratchpad
        }
        output = self.llm_chain.predict(callbacks=callbacks, **inputs)
        reflexion = self.output_parser.parse(output)
        
        # Save reflexion history
        self.trial_history.append(current_trial)
        self.trial_reflexions.append(reflexion)

        return reflexion

        
class ReactReflexionOutputParser(ReflexionOutputParser):
    def parse(self, text: str) -> str:
        # The Reflexion prompt asks the LLM to complete after "New plan: ",
        # so the entire result is the reflexion
        return text