"""Callback Handler streams callback on new llm token in last agent response."""
import warnings
from queue import Queue
from typing import Any, Callable, Iterator, List, Optional, Type, Union

from langchain.agents.agent_types import AgentType
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentFinish, OutputParserException


class StreamingLastResponseCallbackHandler(BaseCallbackHandler):
    """Callback handler for last response streaming in agents.
    Only works with agents using LLMs that support streaming.

    Only the final output of the agent will be streamed.

    Example:
        .. code-block:: python

            from langchain.agents import load_tools, initialize_agent, AgentType
            from langchain.llms import OpenAI
            from langchain.callbacks import StreamingLastResponseCallbackHandler

            llm = OpenAI(temperature=0, streaming=True)
            tools = load_tools(["serpapi", "llm-math"], llm=llm)
            agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

    Usage 1: Callback function to print the next token
        .. code-block:: python

            stream = StreamingLastResponseCallbackHandler.from_agent_type(agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
            @stream.on_last_response_new_token()
            def on_new_token(token: str):
                if token is StopIteration:
                    print("\n[Done]")
                    return
                else:
                    print(token, end="", flush=True)

            agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?", callbacks=[stream])

    Usage 2: Use as iterator
        .. code-block:: python
            import threading

            stream = StreamingLastResponseCallbackHandler.from_agent_type(agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
            def _run():
                agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?", callbacks=[stream])
            threading.Thread(target=_run).start()

            for token in stream:
                print(token, end="", flush=True)

    Usage 3: Post process on-the-fly
        .. code-block:: python
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")

            stream = StreamingLastResponseCallbackHandler.from_agent_type(agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

            @stream.postprocess(sliding_window_step=1, window_size=3)
            def postprocess_func(tokens: List[str]) -> List[str]:
                sentence = "".join(tokens).replace("Python", "LangChain")
                words = [enc.decode([t]) for t in enc.encode(sentence)]  # postprocess output can have different size!
                return words

            def _run():
                agent.run("Is python good?", callbacks=[stream])
            threading.Thread(target=_run).start()

            for token in stream:
                print(token, end="", flush=True)
    """

    def __init__(
        self,
        answer_prefix_phrases: List[str] = ["Final Answer:"],
        error_stop_streaming_phrases: List[str] = [],
        case_sensitive_matching: bool = False,
        output_stream_prefix: bool = False,
        tokenizer: Union[str, Any] = "cl100k_base",
    ) -> None:
        """
        Args:
            answer_prefix_phrases: List of phrases that indicate that the next
                token is the final answer. Multiple phrases are allowed, the first
                one that matches will be used. Phrase matching is case sensitive.
                Some phrases can be a substring of the other phrase. If multiple
                phrases are detected, the longest one will be used.
                Example: ["Final Answer:", "Final Answer"]
            error_stop_streaming_phrases: List of phrases that indicate that the
                next token is an error message and that the streaming should stop.
                Multiple phrases are allowed, the first one that matches will stop
                the streaming. Phrase matching is case sensitive.
            case_sensitive_matching: If True, the answer_prefix_phrases and
                error_stop_streaming_phrases will be matched case sensitive.
                Default: False
            output_stream_prefix: If True, the output stream will include the
                found answer_prefix_phrases. If False, the output stream will
                only include the final answer and exclude the matched
                answer_prefix_phrases. Default: False
            tokenizer: The tokenizer to calculate token length. Can be either encoding
                to use by `tiktoken` or PreTrainedTokenizerBase from transformers.
                Default: "cl100k_base"
        """
        super().__init__()

        if isinstance(tokenizer, str):
            try:
                import tiktoken
            except ImportError:
                raise ImportError(
                    "Could not import tiktoken python package. "
                    "This is needed in order to calculate detection_windows_size for StreamingLastResponseCallbackHandler"
                    "Please install it with `pip install tiktoken`."
                )
            tokenizer = tiktoken.get_encoding(tokenizer)
        else:
            try:
                from transformers import PreTrainedTokenizerBase

                if not isinstance(tokenizer, PreTrainedTokenizerBase):
                    raise ValueError(
                        "Tokenizer received was neither a string nor a PreTrainedTokenizerBase from transformers."
                    )
            except ImportError:
                raise ValueError(
                    "Could not import transformers python package. "
                    "Please install it with `pip install transformers`."
                )

        def _huggingface_tokenizer_length(text: str) -> int:
            return len(tokenizer.encode(text))

        self._get_length_in_tokens = _huggingface_tokenizer_length

        self.case_sensitive_matching = case_sensitive_matching

        self.answer_prefix_phrases = answer_prefix_phrases
        self.error_stop_streaming_phrases = error_stop_streaming_phrases

        # do not use Queue(maxsize=...), because it will block the queue.
        self.detection_queue_size = 1  # using setter below

        # detection_queue will be used to detect the answer_prefix_phrases, error_stop_streaming_phrases, and postprocessing on-the-fly
        self.detection_queue: Queue[str] = Queue()

        # output_queue will be used to stream the output through __iter__()
        self.output_queue: Queue[
            Union[str, Type[StopIteration], OutputParserException]
        ] = Queue()

        # If the answer is reached, the streaming will be started.
        self.is_streaming_answer: bool = False

        self.postprocess_sliding_window_step: int = 1
        self.step_counter: int = 0
        self.output_stream_prefix: bool = output_stream_prefix

        self.callback_func: Callable[
            [Union[str, Type[StopIteration]]], None
        ] = lambda new_token: None
        self.postprocess_func: Optional[Callable[[List[str]], List[str]]] = None

    @property
    def answer_prefix_phrases(self) -> List[str]:
        return self._answer_prefix_phrases

    @answer_prefix_phrases.setter
    def answer_prefix_phrases(self, value: List[str]) -> None:
        """
        Answer prefix phrases should always be sorted by length, so that the longest phrase will be detected first.
        """
        if not value:
            raise ValueError("answer_prefix_phrases cannot be empty.")
        self._answer_prefix_phrases = sorted(value, key=len, reverse=True)
        if not self.case_sensitive_matching:
            self._answer_prefix_phrases = [
                _answer_prefix_phrase.lower()
                for _answer_prefix_phrase in self._answer_prefix_phrases
            ]

    @property
    def error_stop_streaming_phrases(self) -> List[str]:
        return self._error_stop_streaming_phrases

    @error_stop_streaming_phrases.setter
    def error_stop_streaming_phrases(self, value: List[str]) -> None:
        """
        Error stop streaming phrases should always be sorted by length, so that the more informative phrase will be detected first.
        """
        self._error_stop_streaming_phrases = sorted(value, key=len, reverse=True)
        if not self.case_sensitive_matching:
            self._error_stop_streaming_phrases = [
                _error_stop_streaming_phrase.lower()
                for _error_stop_streaming_phrase in self._error_stop_streaming_phrases
            ]

    @property
    def detection_queue_size(self) -> int:
        return self._detection_queue_size

    @detection_queue_size.setter
    def detection_queue_size(self, value: int) -> None:
        """
        The detection queue size is the maximum number of tokens that will be
        stored in the detection queue (soft limit). If the detection queue is full, the
        oldest token will be removed from the queue. The detection queue size
        should be at least the maximum length of the answer_prefix_phrases and
        error_stop_streaming_phrases.
        """
        __max_answer_prefix_phrases_token_len = [
            self._get_length_in_tokens(_answer_prefix_phrase)
            for _answer_prefix_phrase in self._answer_prefix_phrases
        ]
        __max_error_stop_streaming_phrases_token_len = [
            self._get_length_in_tokens(_error_stop_streaming_phrase)
            for _error_stop_streaming_phrase in self._error_stop_streaming_phrases
        ]

        self._detection_queue_size: int = max(
            *__max_answer_prefix_phrases_token_len,
            *__max_error_stop_streaming_phrases_token_len,
            value,
        )

    def __iter__(self) -> Iterator[str]:
        """
        This function is used when the callback handler is used as an iterator.
        """
        while True:
            # Pop out the output queue. If the output queue is empty, it will wait until the output queue is not empty.
            token = self.output_queue.get()

            if token is StopIteration:
                break
            elif isinstance(token, Exception):
                raise token
            elif isinstance(token, str):
                yield token
            else:
                raise TypeError(f"Unknown type: {type(token)} | {token}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """
        This function is called when the agent finishes. It will flush the detection queue when there are no more tokens from on_llm_new_token.
        """
        super().on_agent_finish(finish, **kwargs)
        self._flush_detection_queue()
        self.is_streaming_answer = False

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """
        This function is called when a new token is generated by the LLM. The new token will be put into the detection queue.
        If the detection queue is full, the oldest token will be removed from the queue.
        """
        self.step_counter += 1
        self.detection_queue.put(token)

        if self.is_streaming_answer:
            # if the answer is reached, the streaming will be started.
            last_token = None
            if self.detection_queue.qsize() > self.detection_queue_size:
                if self.step_counter % self.postprocess_sliding_window_step == 0:
                    self._post_process_detection_queue()
                    self._check_abnormal_in_detection_queue()
                last_token = self.detection_queue.get()
                self._callback(last_token)

        elif self.detection_queue.qsize() > self.detection_queue_size:
            # if the answer is not reached, the detection queue will be checked.
            _answer_prefix_phrase = self._check_if_answer_reached()
            # remove all answer prefix tokens from the detection queue
            if _answer_prefix_phrase is not None:
                # In `text-davinci-003` model, token counting is mismatched because OpenAI return "\nFinal Answer:" as a single token during streaming.
                # Therefore, we need to remove the last two tokens from the detection queue to match the token counting.
                # for _ in range(self._get_length_in_tokens(_answer_prefix_phrase) + (0 if "Final Answer" in _answer_prefix_phrase else -2)):
                for _ in range(self._get_length_in_tokens(_answer_prefix_phrase)):
                    _token = self.detection_queue.get()
                    if self.output_stream_prefix:
                        # output the answer prefix token
                        self._callback(_token)
            else:
                # if the answer is not reached, the detection queue will pop out the oldest token.
                self.detection_queue.get()

    def postprocess(
        self,
        sliding_window_step: int = 1,
        window_size: Optional[int] = None,
    ) -> Callable[[Callable[[List[str]], List[str]]], Callable[[List[str]], List[str]]]:
        """
        Decorator to use as postprocess function.

        Args:
            sliding_window_step: Default is 1. This means that the postprocess_func will be applied to the detection queue after every new token.
            window_size: The window size to use for the postprocess_func. The actual used window size will be the maximum of window_size, max length of answer_prefix_phrases, and error_stop_streaming_phrases.
        """

        def _decorator(
            postprocess_func: Callable[[List[str]], List[str]]
        ) -> Callable[[List[str]], List[str]]:
            self.postprocess_func = postprocess_func
            self.postprocess_sliding_window_step = sliding_window_step
            self.detection_queue_size = max(
                self.detection_queue_size,
                window_size or 1,
            )
            return postprocess_func

        return _decorator

    def on_last_response_new_token(
        self,
    ) -> Callable[
        [Callable[[Union[str, Type[StopIteration]]], None]],
        Callable[[Union[str, Type[StopIteration]]], None],
    ]:
        """
        Decorator to use as callback function.
        """

        def _decorator(
            callback_func: Callable[[Union[str, Type[StopIteration]]], None]
        ) -> Callable[[Union[str, Type[StopIteration]]], None]:
            self.callback_func = callback_func
            return callback_func

        return _decorator

    @classmethod
    def from_agent_type(
        cls,
        agent: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        **kwargs: Any,
    ) -> "StreamingLastResponseCallbackHandler":
        """
        Create a callback handler for last response streaming for a specific agent type.
        For custom agent, please use the constructor directly.
        """
        if agent == AgentType.ZERO_SHOT_REACT_DESCRIPTION:
            return cls(
                answer_prefix_phrases=[
                    "Final Answer:",
                ],
                **kwargs,
            )
        elif agent == AgentType.CONVERSATIONAL_REACT_DESCRIPTION:
            return cls(
                answer_prefix_phrases=[
                    "Do I need to use a tool? No\nAI:",
                    "Do I need to use a tool? No",
                ],
                error_stop_streaming_phrases=[
                    "Do I need to use a tool? No\nAction:",
                ],
                **kwargs,
            )

        elif agent == AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION:
            # TODO: Post processing remove last '"\n}' after final answer
            raise NotImplementedError
            return cls(
                answer_prefix_phrases=[
                    'Final Answer",\n    "action_input": "',
                    'Final Answer",\n  "action_input": "',
                ],
                **kwargs,
            )
        elif agent == AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION:
            # TODO: Post processing remove last '"\n}\n```' after final answer
            raise NotImplementedError
            return cls(
                answer_prefix_phrases=[
                    'Final Answer",\n    "action_input": "',
                    'Final Answer",\n  "action_input": "',
                ],
                **kwargs,
            )
        else:
            raise NotImplementedError

    def _post_process_detection_queue(self) -> None:
        """
        Post process on-the-fly the detection queue by using used-defined postprocess_func.
        This function will be called every postprocess_sliding_window_step.
        """
        if self.postprocess_func is not None:
            tokens = list(self.detection_queue.queue)
            tokens = self.postprocess_func(tokens)
            self.detection_queue.queue.clear()
            for token in tokens:
                self.detection_queue.put(token)

    def _check_abnormal_in_detection_queue(self) -> None:
        """
        Check if the detection queue is abnormal. If the detection queue is abnormal, it will raise OutputParserException and stop the streaming.
        Check by using error_stop_streaming_phrases. If the error_stop_streaming_phrases is detected, the streaming will be stopped.
        """
        sentence = "".join(self.detection_queue.queue)

        for error_stop_streaming_phrases in self.error_stop_streaming_phrases:
            if error_stop_streaming_phrases in sentence:
                self._callback(
                    OutputParserException(
                        f"Abnormal in detection queue detected. Detection queue: '{self.detection_queue.queue}'. Abnormal: '{error_stop_streaming_phrases}'"
                    )
                )

    def _flush_detection_queue(self) -> None:
        """
        Flush detection queue. This will be called when the agent is finished to flush all the remaining tokens in the detection queue.
        """
        while not self.detection_queue.empty():
            if not self.is_streaming_answer:
                _answer_prefix_phrase = self._check_if_answer_reached()
                if _answer_prefix_phrase is not None:
                    # remove all answer prefix tokens from detection queue
                    if not self.output_stream_prefix:
                        for _ in range(self._get_length_in_tokens(_answer_prefix_phrase)):
                            while self.detection_queue.queue[0] == "":
                                self.detection_queue.get()
                            self.detection_queue.get()
                    else:
                        for _ in range(self._get_length_in_tokens(_answer_prefix_phrase)):
                            self._callback(self.detection_queue.get())
                else:
                    self.detection_queue.get()
            else:
                self._callback(self.detection_queue.get())

        if not self.is_streaming_answer:
            warnings.warn(
                "StreamingLastResponseCallbackHandler is not streaming answer, but agent_finish is called."
            )

        self._callback(StopIteration)

    def _callback(
        self, text: Union[str, OutputParserException, Type[StopIteration]]
    ) -> None:
        """
        Callback function. It will put the text to the output queue, and call the user-defined callback function.
        """
        if isinstance(text, OutputParserException):
            self.output_queue.put(text)
            raise text
        elif text is StopIteration:
            self.output_queue.put(text)
            self.callback_func(text)
        elif isinstance(text, str):
            self.output_queue.put(text)
            self.callback_func(text)
        else:
            raise TypeError(f"Unknown type: {type(text)} | {text}")

    def _check_if_answer_reached(self) -> Optional[str]:
        """
        Check if the answer is reached. If the answer is reached, it will return the answer prefix phrase.
        If the answer is not reached, it will return None.
        """
        if self.detection_queue.queue[0] == "":
            return None
        for _answer_prefix_str in self.answer_prefix_phrases:
            current_output = "".join(self.detection_queue.queue).strip()

            if not self.case_sensitive_matching:
                current_output = current_output.lower()

            if current_output.startswith(_answer_prefix_str):
                self.is_streaming_answer = True
                return _answer_prefix_str
        return None
