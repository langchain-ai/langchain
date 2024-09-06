# -*- coding : utf-8 -*-
# @Time      :2024-09-02 15:05
# @Author   : zy(子永)
# @ Software: Pycharm - windows
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage

from typing import Any, Dict, Iterator, List, Optional, Union, Type

from openai import OpenAI
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.runnables import Runnable
from pydantic import BaseModel


class SFLLM(LLM):
    """
    基于siliconflow的LLM实现。
    参考链接：https://docs.siliconflow.cn/docs/%E5%85%B3%E4%BA%8Esiliconcloud
    """
    api_key: str
    base_url = "https://api.siliconflow.cn/v1"
    client: OpenAI = None

    def with_structured_output(self, schema: Union[Dict, Type[BaseModel]], **kwargs: Any) -> Runnable[
        LanguageModelInput, Union[Dict, BaseModel]]:
        pass

    def _sf_chat(self, messages, model='alibaba/Qwen2-7B-Instruct', stream=True) -> str:
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream
        )
        res_str = ""
        for chunk in response:
            _content = chunk.choices[0].delta.content
            if _content:
                res_str += chunk.choices[0].delta.content
        return res_str

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        ans = self._sf_chat([{"role": "user", "content": prompt}], **kwargs)
        return ans

    def _stream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the LLM on the given prompt.

        This method should be overridden by subclasses that support streaming.

        If not implemented, the default behavior of calls to stream will be to
        fallback to the non-streaming version of the model and return
        the output as a single chunk.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An iterator of GenerationChunks.
        """
        for char in prompt[: self.n]:
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

            yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": "CustomChatModel",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"


if __name__ == '__main__':
    model = SFLLM(api_key="sk-your-api-key")
    result = model.invoke([HumanMessage(content="1+1")])
    print(result)
