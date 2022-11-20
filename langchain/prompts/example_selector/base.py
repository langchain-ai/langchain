"""Interface for selecting examples to include in prompts."""
from abc import ABC, abstractmethod
from typing import Dict, List


class BaseExampleSelector(ABC):
    """Interface for selecting examples to include in prompts."""

    @abstractmethod
    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs."""
