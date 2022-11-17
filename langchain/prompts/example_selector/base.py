from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseExampleSelector(ABC):
    @abstractmethod
    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs."""
