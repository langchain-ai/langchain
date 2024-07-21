from abc import abstractmethod, ABC
from typing import Dict, Optional

from pydantic import BaseModel


class ModelConfig(BaseModel, ABC):
    model: str
    batch_size: int
    dimension: int

    @abstractmethod
    def get_query_parameters(self) -> Dict:
        """Provide default parameters for query; must be overridden in subclass."""
        pass

    @abstractmethod
    def get_doc_parameters(self) -> Dict:
        """Provide default parameters for documents; must be overridden in subclass."""
        pass


class MultilingualE5LargeConfig(ModelConfig):
    model = "multilingual-e5-large"
    batch_size = 96
    dimension = 1024
    truncation: Optional[str] = None

    def get_query_parameters(self) -> Dict:
        return {"input_type": "passage", "truncation": self.truncation}

    def get_doc_parameters(self) -> Dict:
        return {"input_type": "passage", "truncation": self.truncation}
