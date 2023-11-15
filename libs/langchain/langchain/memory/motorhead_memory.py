from typing import Any, Dict, List, Optional

import requests

from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema.messages import get_buffer_string

MANAGED_URL = "https://api.getmetal.io/v1/motorhead"
# LOCAL_URL = "http://localhost:8080"


class MotorheadMemory(BaseChatMemory):
    """Chat message memory backed by Motorhead service."""

    url: str = MANAGED_URL
    timeout: int = 3000
    memory_key: str = "history"
    session_id: str
    context: Optional[str] = None

    # Managed Params
    api_key: Optional[str] = None
    client_id: Optional[str] = None

    def __get_headers(self) -> Dict[str, str]:
        is_managed = self.url == MANAGED_URL

        headers = {
            "Content-Type": "application/json",
        }

        if is_managed and not (self.api_key and self.client_id):
            raise ValueError(
                """
                You must provide an API key or a client ID to use the managed
                version of Motorhead. Visit https://getmetal.io for more information.
                """
            )

        if is_managed and self.api_key and self.client_id:
            headers["x-metal-api-key"] = self.api_key
            headers["x-metal-client-id"] = self.client_id

        return headers

    async def init(self) -> None:
        res = requests.get(
            f"{self.url}/sessions/{self.session_id}/memory",
            timeout=self.timeout,
            headers=self.__get_headers(),
        )
        res_data = res.json()
        res_data = res_data.get("data", res_data)  # Handle Managed Version

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
            headers=self.__get_headers(),
        )
        super().save_context(inputs, outputs)

    def delete_session(self) -> None:
        """Delete a session"""
        requests.delete(f"{self.url}/sessions/{self.session_id}/memory")
