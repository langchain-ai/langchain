import time
from typing import Any, Dict, List, Optional, cast

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult


def import_infino() -> Any:
    """Import the infino client."""
    try:
        from infinopy import InfinoClient
    except ImportError:
        raise ImportError(
            "To use the Infino callbacks manager you need to have the"
            " `infinopy` python package installed."
            "Please install it with `pip install infinopy`"
        )
    return InfinoClient()


def import_tiktoken() -> Any:
    """Import tiktoken for counting tokens for OpenAI models."""
    try:
        import tiktoken
    except ImportError:
        raise ImportError(
            "To use the ChatOpenAI model with Infino callback manager, you need to "
            "have the `tiktoken` python package installed."
            "Please install it with `pip install tiktoken`"
        )
    return tiktoken


def get_num_tokens(string: str, openai_model_name: str) -> int:
    """Calculate num tokens for OpenAI with tiktoken package.

    Official documentation: https://github.com/openai/openai-cookbook/blob/main
                            /examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    tiktoken = import_tiktoken()

    encoding = tiktoken.encoding_for_model(openai_model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


class InfinoCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs to Infino."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        # Set Infino client
        self.client = import_infino()
        self.model_id = model_id
        self.model_version = model_version
        self.verbose = verbose
        self.is_chat_openai_model = False
        self.chat_openai_model_name = "gpt-3.5-turbo"

    def _send_to_infino(
        self,
        key: str,
        value: Any,
        is_ts: bool = True,
    ) -> None:
        """Send the key-value to Infino.

        Parameters:
        key (str): the key to send to Infino.
        value (Any): the value to send to Infino.
        is_ts (bool): if True, the value is part of a time series, else it
                      is sent as a log message.
        """
        payload = {
            "date": int(time.time()),
            key: value,
            "labels": {
                "model_id": self.model_id,
                "model_version": self.model_version,
            },
        }
        if self.verbose:
            print(f"Tracking {key} with Infino: {payload}")

        # Append to Infino time series only if is_ts is True, otherwise
        # append to Infino log.
        if is_ts:
            self.client.append_ts(payload)
        else:
            self.client.append_log(payload)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Log the prompts to Infino, and set start time and error flag."""
        for prompt in prompts:
            self._send_to_infino("prompt", prompt, is_ts=False)

        # Set the error flag to indicate no error (this will get overridden
        # in on_llm_error if an error occurs).
        self.error = 0

        # Set the start time (so that we can calculate the request
        # duration in on_llm_end).
        self.start_time = time.time()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Do nothing when a new token is generated."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Log the latency, error, token usage, and response to Infino."""
        # Calculate and track the request latency.
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        self._send_to_infino("latency", duration)

        # Track success or error flag.
        self._send_to_infino("error", self.error)

        # Track prompt response.
        for generations in response.generations:
            for generation in generations:
                self._send_to_infino("prompt_response", generation.text, is_ts=False)

        # Track token usage (for non-chat models).
        if (response.llm_output is not None) and isinstance(response.llm_output, Dict):
            token_usage = response.llm_output["token_usage"]
            if token_usage is not None:
                prompt_tokens = token_usage["prompt_tokens"]
                total_tokens = token_usage["total_tokens"]
                completion_tokens = token_usage["completion_tokens"]
                self._send_to_infino("prompt_tokens", prompt_tokens)
                self._send_to_infino("total_tokens", total_tokens)
                self._send_to_infino("completion_tokens", completion_tokens)

        # Track completion token usage (for openai chat models).
        if self.is_chat_openai_model:
            messages = " ".join(
                cast(str, cast(ChatGeneration, generation).message.content)
                for generation in generations
            )
            completion_tokens = get_num_tokens(
                messages, openai_model_name=self.chat_openai_model_name
            )
            self._send_to_infino("completion_tokens", completion_tokens)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Set the error flag."""
        self.error = 1

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Do nothing when LLM chain starts."""
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Do nothing when LLM chain ends."""
        pass

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Need to log the error."""
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Do nothing when tool starts."""
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Do nothing when agent takes a specific action."""
        pass

    def on_tool_end(
        self,
        output: str,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Do nothing when tool ends."""
        pass

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """Do nothing when tool outputs an error."""
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Do nothing."""
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Do nothing."""
        pass

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts running."""

        # Currently, for chat models, we only support input prompts for ChatOpenAI.
        # Check if this model is a ChatOpenAI model.
        values = serialized.get("id")
        if values:
            for value in values:
                if value == "ChatOpenAI":
                    self.is_chat_openai_model = True
                    break

        # Track prompt tokens for ChatOpenAI model.
        if self.is_chat_openai_model:
            invocation_params = kwargs.get("invocation_params")
            if invocation_params:
                model_name = invocation_params.get("model_name")
                if model_name:
                    self.chat_openai_model_name = model_name
                    prompt_tokens = 0
                    for message_list in messages:
                        message_string = " ".join(
                            cast(str, msg.content) for msg in message_list
                        )
                        num_tokens = get_num_tokens(
                            message_string,
                            openai_model_name=self.chat_openai_model_name,
                        )
                        prompt_tokens += num_tokens

                    self._send_to_infino("prompt_tokens", prompt_tokens)

        if self.verbose:
            print(
                f"on_chat_model_start: is_chat_openai_model= \
                  {self.is_chat_openai_model}, \
                  chat_openai_model_name={self.chat_openai_model_name}"
            )

        # Send the prompt to infino
        prompt = " ".join(
            cast(str, msg.content) for sublist in messages for msg in sublist
        )
        self._send_to_infino("prompt", prompt, is_ts=False)

        # Set the error flag to indicate no error (this will get overridden
        # in on_llm_error if an error occurs).
        self.error = 0

        # Set the start time (so that we can calculate the request
        # duration in on_llm_end).
        self.start_time = time.time()
