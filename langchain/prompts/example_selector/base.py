from abc import ABC, abstractmethod
from typing import Any, List


class BaseExampleSelector(ABC):

    @abstractmethod
    def select_examples(self, **kwargs: Any) -> List[dict]:
        """Select which examples to use based on the inputs."""

