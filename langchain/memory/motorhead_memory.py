import os
import json
from typing import List, Dict, Any, Optional
import requests
from chat_memory import BaseChatMemory

MOTORHEAD_URL = os.environ.get("MOTORHEAD_URL", "http://localhost:8080")

class MotorheadMemoryMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

class MotorheadMemoryInput:
    def __init__(self, chat_history, return_messages: bool, session_id: str, input_key: Optional[str] = None, output_key: Optional[str] = None):
        self.chat_history = chat_history
        self.return_messages = return_messages
        self.session_id = session_id
        self.input_key = input_key
        self.output_key = output_key

class MotorheadMemory(BaseChatMemory):
    def __init__(self, fields: Optional[MotorheadMemoryInput] = None):
        if fields is None:
            fields = MotorheadMemoryInput(None, False, '')

        super().__init__(return_messages=fields.return_messages, input_key=fields.input_key, output_key=fields.output_key, chat_history=fields.chat_history)

        self.motorhead_url = MOTORHEAD_URL
        self.timeout = 3000
        self.memory_key = "history"
        self.session_id = fields.session_id
        self.context = None

    async def init(self):
        res = requests.get(f"{MOTORHEAD_URL}/sessions/{self.session_id}/memory", timeout=self.timeout, headers={"Content-Type": "application/json"})
        res_data = res.json()
        messages = res_data.get("messages", [])
        context = res_data.get("context", "NONE")

        for message in messages:
            if message["role"] == "AI":
                self.chat_history.add_ai_chat_message(message["content"])
            else:
                self.chat_history.add_user_message(message["content"])

        if context and context != "NONE":
            self.context = context

    def load_memory_variables(self, values):
        if self.return_messages:
            return {self.memory_key: self.chat_history.messages}
        else:
            return {self.memory_key: get_buffer_string(self.chat_history.messages)}

    def save_context(self, input_values, output_values):
        requests.post(f"{MOTORHEAD_URL}/sessions/{self.session_id}/memory", timeout=self.timeout, json={
            "messages": [
                {"role": "Human", "content": f"{input_values['input']}"},
                {"role": "AI", "content": f"{output_values['response']}"}
            ]
        }, headers={"Content-Type": "application/json"})

        super().save_context(input_values, output_values)
