from typing import Any, Dict, List, Optional

import requests

from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import get_buffer_string


class MotorheadMemory(BaseChatMemory):
    url: str = "http://localhost:8080"
    timeout = 3000
    memory_key = "history"
    session_id: str
    context: Optional[str] = None

    async def init(self) -> None:
        res = requests.get(
            f"{self.url}/sessions/{self.session_id}/memory",
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
        )
        res_data = res.json()
        messages = res_data.get("messages", [])
        context = res_data.get("context", "NONE")

        for message in reversed(messages):
            if message["role"] == "AI":
                self.chat_memory.add_ai_message(message["content"])
            else:
                self.chat_memory.add_user_message(message["content"])

        if context and context != "NONE":
            self.context = context

    def load_memory_variables(self, values: Dict[str, Any]) -> Dict[str, Any]:
        if self.return_messages:
            return {self.memory_key: self.chat_memory.messages}
        else:
            return {self.memory_key: get_buffer_string(self.chat_memory.messages)}

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        input_str, output_str = self._get_input_output(inputs, outputs)
        requests.post(
            f"{self.url}/sessions/{self.session_id}/memory",
            timeout=self.timeout,
            json={
                "messages": [
                    {"role": "Human", "content": f"{input_str}"},
                    {"role": "AI", "content": f"{output_str}"},
                ]
            },
            headers={"Content-Type": "application/json"},
        )
        super().save_context(inputs, outputs)
