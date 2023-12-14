"""Chat Model Components Derived from ChatModel/NVAIPlay"""
from __future__ import annotations

import base64
import logging
import os
import urllib.parse
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import requests
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage, ChatMessage, ChatMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from langchain_nvidia_aiplay import _common as nv_aiplay

logger = logging.getLogger(__name__)


def _is_openai_parts_format(part: dict) -> bool:
    return "type" in part


def _is_url(s: str) -> bool:
    try:
        result = urllib.parse.urlparse(s)
        return all([result.scheme, result.netloc])
    except Exception as e:
        logger.debug(f"Unable to parse URL: {e}")
        return False


def _is_b64(s: str) -> bool:
    return s.startswith("data:image")


def _url_to_b64_string(image_source: str) -> str:
    b64_template = "data:image/png;base64,{b64_string}"
    try:
        if _is_url(image_source):
            response = requests.get(image_source)
            response.raise_for_status()
            encoded = base64.b64encode(response.content).decode("utf-8")
            return b64_template.format(b64_string=encoded)
        elif _is_b64(image_source):
            return image_source
        elif os.path.exists(image_source):
            with open(image_source, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                return b64_template.format(b64_string=encoded)
        else:
            raise ValueError(
                "The provided string is not a valid URL, base64, or file path."
            )
    except Exception as e:
        raise ValueError(f"Unable to process the provided image source: {e}")


class ChatNVAIPlay(nv_aiplay._NVAIPlayClient, SimpleChatModel):
    """NVAIPlay chat model.

    Example:
        .. code-block:: python

            from langchain_nvidia_aiplay import ChatNVAIPlay


            model = ChatNVAIPlay(model="llama2_13b")
            response = model.invoke("Hello")
    """

    @property
    def _llm_type(self) -> str:
        """Return type of NVIDIA AI Playground Interface."""
        return "chat-nvidia-ai-playground"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[Sequence[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        labels: Optional[dict] = None,
        **kwargs: Any,
    ) -> str:
        """Invoke on a single list of chat messages."""
        inputs = self.custom_preprocess(messages)
        responses = self.get_generation(
            inputs=inputs, stop=stop, labels=labels, **kwargs
        )
        outputs = self.custom_postprocess(responses)
        return outputs

    def _get_filled_chunk(
        self, text: str, role: Optional[str] = "assistant"
    ) -> ChatGenerationChunk:
        """Fill the generation chunk."""
        return ChatGenerationChunk(message=ChatMessageChunk(content=text, role=role))

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[Sequence[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        labels: Optional[dict] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Allows streaming to model!"""
        inputs = self.custom_preprocess(messages)
        for response in self.get_stream(
            inputs=inputs, stop=stop, labels=labels, **kwargs
        ):
            chunk = self._get_filled_chunk(self.custom_postprocess(response))
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[Sequence[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        labels: Optional[dict] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        inputs = self.custom_preprocess(messages)
        async for response in self.get_astream(
            inputs=inputs, stop=stop, labels=labels, **kwargs
        ):
            chunk = self._get_filled_chunk(self.custom_postprocess(response))
            yield chunk
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    def custom_preprocess(
        self, msg_list: Sequence[BaseMessage]
    ) -> List[Dict[str, str]]:
        # The previous author had a lot of custom preprocessing here
        # but I'm just going to assume it's a list
        return [self.preprocess_msg(m) for m in msg_list]

    def _process_content(self, content: Union[str, List[Union[dict, str]]]) -> str:
        if isinstance(content, str):
            return content
        string_array: list = []

        for part in content:
            if isinstance(part, str):
                string_array.append(part)
            elif isinstance(part, Mapping):
                # OpenAI Format
                if _is_openai_parts_format(part):
                    if part["type"] == "text":
                        string_array.append(str(part["text"]))
                    elif part["type"] == "image_url":
                        img_url = part["image_url"]
                        if isinstance(img_url, dict):
                            if "url" not in img_url:
                                raise ValueError(
                                    f"Unrecognized message image format: {img_url}"
                                )
                            img_url = img_url["url"]
                        b64_string = _url_to_b64_string(img_url)
                        string_array.append(f'<img src="{b64_string}" />')
                    else:
                        raise ValueError(
                            f"Unrecognized message part type: {part['type']}"
                        )
                else:
                    raise ValueError(f"Unrecognized message part format: {part}")
        return "".join(string_array)

    def preprocess_msg(self, msg: BaseMessage) -> Dict[str, str]:
        ## (WFH): Previous author added a bunch of
        # custom processing here, but I'm just going to support
        # the LCEL api.
        if isinstance(msg, BaseMessage):
            role_convert = {"ai": "assistant", "human": "user"}
            if isinstance(msg, ChatMessage):
                role = msg.role
            else:
                role = msg.type
            role = role_convert.get(role, role)
            content = self._process_content(msg.content)
            return {"role": role, "content": content}
        raise ValueError(f"Invalid message: {repr(msg)} of type {type(msg)}")

    def custom_postprocess(self, msg: dict) -> str:
        if "content" in msg:
            return msg["content"]
        logger.warning(
            f"Got ambiguous message in postprocessing; returning as-is: msg = {msg}"
        )
        return str(msg)
