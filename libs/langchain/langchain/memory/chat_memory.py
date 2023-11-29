import json
import warnings
from abc import ABC
from typing import Any, Dict, Optional, Tuple

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.memory import BaseMemory
from langchain_core.pydantic_v1 import Field

from langchain.load.load import loads
from langchain.load.serializable import Serializable
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.memory.utils import get_prompt_input_key
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import SystemMessage


class BaseChatMemory(BaseMemory, Serializable, ABC):
    """Abstract base class for chat memory."""

    chat_memory: BaseChatMessageHistory = Field(default_factory=ChatMessageHistory)
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    return_messages: bool = False

    def _get_input_output(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> Tuple[str, str]:
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key
        return inputs[prompt_input_key], outputs[output_key]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_user_message(input_str)
        self.chat_memory.add_ai_message(output_str)

    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_memory.clear()

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Is this class serializable?"""
        return True

    def to_json(self) -> Dict[str, Any]:
        serialized = super().to_json()

        # Use the toJSON method from chat_memory to get string representation
        chat_memory_obj = self.chat_memory.to_json()

        chat_memory_dict = chat_memory_obj

        self_dict = {"chat_memory": chat_memory_dict}

        self_dict.update(
            {key: value for key, value in vars(self).items() if key != "chat_memory"}
        )

        serialized["obj"] = json.loads(
            json.dumps(
                self_dict,
                default=lambda o: custom_serializer(o),
                sort_keys=True,
                indent=4,
            )
        )

        if (serialized["kwargs"]).get("llm"):
            serialized["kwargs"]["llm"] = type(serialized["kwargs"]["llm"]).__name__
        return serialized

    @classmethod
    def from_json(cls, json_input: str, llm: BaseLanguageModel = None) -> Any:
        memory_dict = json.loads(json_input)

        if memory_dict.get("id"):
            if cls.__name__ != memory_dict["id"][-1]:
                raise ValueError(
                    f"Memory object type {cls.__name__} passed differs from\
                                  type in json {memory_dict['id'][-1]}"
                )
        if memory_dict.get("obj") and (memory_dict["obj"]).get("llm"):
            if type(llm).__name__ != memory_dict["obj"]["llm"]["id"][-1]:
                warnings.warn(
                    f"llm provided is different from llm in\
                              json: {memory_dict['obj']['llm']['repr']}"
                )

            del memory_dict["obj"]["llm"]

        if memory_dict.get("kwargs") and (memory_dict["kwargs"]).get("llm"):
            del memory_dict["kwargs"]["llm"]

        deserialized = (
            loads(json.dumps(memory_dict), llm=llm)
            if llm is not None
            else loads(json.dumps(memory_dict))
        )

        chat_memory = BaseChatMessageHistory.from_json(
            json.dumps(memory_dict["obj"]["chat_memory"])
        )

        # Extract additional attributes from memory_dict
        additional_attributes = {
            key: memory_dict["obj"][key]
            for key in memory_dict["obj"]
            if not (
                key == chat_memory
                or isinstance(getattr(deserialized, key, None), Serializable)
            )
        }
        if (llm is not None) and hasattr(cls, "llm"):
            additional_attributes["llm"] = llm

        if additional_attributes.get("summary_message_cls"):
            del additional_attributes["summary_message_cls"]
            deserialized.summary_message_cls = SystemMessage

        deserialized.chat_memory = chat_memory
        deserialized.__dict__.update(additional_attributes)

        return deserialized


def custom_serializer(obj: Any) -> Any:
    if isinstance(obj, Serializable):
        return obj.to_json()
    elif isinstance(obj, type):
        return obj.__name__
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    else:
        return TypeError(
            "Object of type '%s' is not JSON serializable" % type(obj).__name__
        )
