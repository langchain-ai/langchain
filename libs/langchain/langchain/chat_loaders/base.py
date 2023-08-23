from abc import ABC, abstractmethod
from typing import List, Iterator
from langchain.schema.messages import BaseMessage
from pydantic import BaseModel

class BaseChatLoader(BaseModel, ABC):

    def load_messages(self) -> List[List[BaseMessage]]:
        return list(self.lazy_load_messages())

    @abstractmethod
    def lazy_load_messages(self) -> Iterator[List[BaseMessage]]:
        """Return iterator of messages."""