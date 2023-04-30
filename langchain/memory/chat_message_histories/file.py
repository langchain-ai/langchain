import json
import logging
from pathlib import Path
from typing import List

from pydantic import validator

from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
    messages_from_dict,
    messages_to_dict,
    MessageLog
)

logger = logging.getLogger(__name__)


class FileChatMessageHistory(BaseChatMessageHistory):
    """
    Chat message history that stores history in a local file.

    Args:
        file_path: path of the local file to store the messages.
    """

    file_path: Path

    @validator('file_path')
    def initialize_file_path(cls, value) -> Path:
        if type(value) == str:
            file_path = Path(value)
        
        else:
            file_path = value

        if not file_path.exists():
            file_path.touch()
            file_path.write_text(json.dumps([]))  
        
        return file_path

    def load_message_logs(self) -> List[MessageLog]:
        if not self.file_path.exists():
            self.file_path.touch()
            self.file_path.write_text(json.dumps([]))
            return []

        else:
           items = json.loads(self.file_path.read_text())
           message_logs = [MessageLog(**item) for item in items]
           return message_logs
    
    def save_message_log(self, message_log: MessageLog) -> None:
        current_message_logs = self.load_message_logs()
        current_message_logs.append(message_log)
        current_message_logs = [i.json() for i in current_message_logs]
        # NOTE: Maybe we should just use JSONL?
        current_message_logs = ",".join(current_message_logs)
        self.file_path.write_text('[' + current_message_logs + ']')
    
    def _clear(self) -> None:
        self.file_path.write_text(json.dumps([]))
