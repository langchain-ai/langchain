from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from langchain.chains.base import Chain
from langchain_core.callbacks.manager import CallbackManagerForChainRun

from langchain_experimental.tot.thought import ThoughtValidity


class ToTChecker(Chain, ABC):
    """
    Tree of Thought (ToT) checker.

    This is an abstract ToT checker that must be implemented by the user. You
    can implement a simple rule-based checker or a more sophisticated
    neural network based classifier.
    """

    output_key: str = "validity"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        """The checker input keys.

        :meta private:
        """
        return ["problem_description", "thoughts"]

    @property
    def output_keys(self) -> List[str]:
        """The checker output keys.

        :meta private:
        """
        return [self.output_key]

    @abstractmethod
    def evaluate(
        self,
        problem_description: str,
        thoughts: Tuple[str, ...] = (),
    ) -> ThoughtValidity:
        """
        Evaluate the response to the problem description and return the solution type.
        """

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, ThoughtValidity]:
        return {self.output_key: self.evaluate(**inputs)}
