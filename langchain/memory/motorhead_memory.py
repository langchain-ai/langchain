import os
from pydantic import BaseModel
from typing import Any, Dict, Optional
from langchain.memory.chat_memory import BaseChatMemory


MOTORHEAD_URL = os.environ["MOTORHEAD_URL"] or "http://localhost:8080"


class MotorheadMemory(BaseChatMemory, BaseModel):
    motorhead_url: str = MOTORHEAD_URL
    timeout: int = 3000
    memory_key: str = "history"
    session_id: str
    context: Optional[str] = None


    def __init__(self, **data: Any) -> None:
      super().__init__(**data)
      self.session_id = self.chat_memory.get_session_id()