import boto3
from langchain.chains.base import BaseMemory
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from datetime import datetime

now = datetime.now()

def _get_prompt_input_key(inputs: Dict[str, Any], memory_variables: List[str]) -> str:
    # "stop" is a special key that can be passed as input but is not used to
    # format the prompt.
    prompt_input_keys = list(set(inputs).difference(memory_variables + ["stop"]))
    if len(prompt_input_keys) != 1:
        raise ValueError(f"One input key expected got {prompt_input_keys}")
    return prompt_input_keys[0]

class DynamoDBBuffer:

    def __init__(self, table_name):
        self.table_name = table_name
        self.ddb = boto3.client("dynamodb")

    def read(self, session_id: str) -> str:
        """Retrieve history buffer from DynamoDB."""
        try:
            client = boto3.client("dynamodb")
            response = client.get_item(
                TableName=self.table_name,
                Key={'id': {"S": session_id}}
            )
        except Exception as e:
            print(f"Error loading memory from DynamoDB: {e}")
            buffer = ""
            return buffer
        else:
            item = response.get("Item", {})
            if len(item) == 0:
                buffer = ""
            else:
                buffer = item.get("buffer", {}).get("S", "")
            return buffer

    def write(self, session_id: str, history: str):
        """Save conversation history to DynamoDB."""
        try:
            client = boto3.client("dynamodb")
            client.put_item(
                TableName=self.table_name,
                Item={
                    'id': {"S": session_id},
                    "buffer": {"S": history},
                    "updatedAt": {"S": str(now)}
                }
            )
        except Exception as e:
            print(f"Error saving conversation history to DynamoDB: {e}")

    def clear(self, session_id: str):
        """Clear memory contents from DynamoDB."""
        try:
            client = boto3.client("dynamodb")
            client.delete_item(
                TableName=self.table_name,
                Key={'id': {"S": session_id}}
            )
        except Exception as e:
            print(f"Error clearing memory from DynamoDB: {e}")

class ServerlessMemory(BaseMemory, BaseModel):
    """Buffer for storing conversation memory in DynamoDB."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    """Prefix to use for AI generated responses."""
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    memory_key: str = "chat_history"
    table_name: str = "ConversationHistory"
    parition_key: str = "session_id"
    buffer: DynamoDBBuffer
    session_id: str

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Retrieve history buffer from DynamoDB."""
        response = self.buffer.read(self.session_id)
        return {self.memory_key: response}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to DynamoDB."""
        if self.input_key is None:
            prompt_input_key = _get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key
        human = f"{self.human_prefix}: " + inputs[prompt_input_key]
        ai = f"{self.ai_prefix}: " + outputs[output_key]
        history = self.buffer.read(self.session_id) + "\n".join([human, ai]) + "\n"
        self.buffer.write(self.session_id, history)

    def clear(self) -> None:
        """Clear memory contents from DynamoDB."""
        self.buffer.clear(self.session_id)

