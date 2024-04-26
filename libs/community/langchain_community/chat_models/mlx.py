"""MLX Chat Wrapper."""

from typing import Any, Iterator, List, Optional

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
)

from langchain_community.llms.mlx_pipeline import MLXPipeline

DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful, and honest assistant."""


class ChatMLX(BaseChatModel):
    """
    Wrapper for using MLX LLM's as ChatModels.

    Works with `MLXPipeline` LLM.

    To use, you should have the ``mlx-lm`` python package installed.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import chatMLX
            from langchain_community.llms import MLXPipeline

            llm = MLXPipeline.from_model_id(
                model_id="mlx-community/quantized-gemma-2b-it",
            )
            chat = chatMLX(llm=llm)

    """

    llm: MLXPipeline
    system_message: SystemMessage = SystemMessage(content=DEFAULT_SYSTEM_PROMPT)
    tokenizer: Any = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.tokenizer = self.llm.tokenizer

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm_input = self._to_chat_prompt(messages)
        llm_result = self.llm._generate(
            prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
        )
        return self._to_chat_result(llm_result)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm_input = self._to_chat_prompt(messages)
        llm_result = await self.llm._agenerate(
            prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
        )
        return self._to_chat_result(llm_result)

    def _to_chat_prompt(
        self,
        messages: List[BaseMessage],
        tokenize: bool = False,
        return_tensors: Optional[str] = None,
    ) -> str:
        """Convert a list of messages into a prompt format expected by wrapped LLM."""
        if not messages:
            raise ValueError("At least one HumanMessage must be provided!")

        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("Last message must be a HumanMessage!")

        messages_dicts = [self._to_chatml_format(m) for m in messages]

        return self.tokenizer.apply_chat_template(
            messages_dicts,
            tokenize=tokenize,
            add_generation_prompt=True,
            return_tensors=return_tensors,
        )

    def _to_chatml_format(self, message: BaseMessage) -> dict:
        """Convert LangChain message to ChatML format."""

        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        return {"role": role, "content": message.content}

    @staticmethod
    def _to_chat_result(llm_result: LLMResult) -> ChatResult:
        chat_generations = []

        for g in llm_result.generations[0]:
            chat_generation = ChatGeneration(
                message=AIMessage(content=g.text), generation_info=g.generation_info
            )
            chat_generations.append(chat_generation)

        return ChatResult(
            generations=chat_generations, llm_output=llm_result.llm_output
        )

    @property
    def _llm_type(self) -> str:
        return "mlx-chat-wrapper"

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        import mlx.core as mx
        from mlx_lm.utils import generate_step

        try:
            import mlx.core as mx
            from mlx_lm.utils import generate_step

        except ImportError:
            raise ValueError(
                "Could not import mlx_lm python package. "
                "Please install it with `pip install mlx_lm`."
            )
        model_kwargs = kwargs.get("model_kwargs", self.llm.pipeline_kwargs)
        temp: float = model_kwargs.get("temp", 0.0)
        max_new_tokens: int = model_kwargs.get("max_tokens", 100)
        repetition_penalty: Optional[float] = model_kwargs.get(
            "repetition_penalty", None
        )
        repetition_context_size: Optional[int] = model_kwargs.get(
            "repetition_context_size", None
        )

        llm_input = self._to_chat_prompt(messages, tokenize=True, return_tensors="np")

        prompt_tokens = mx.array(llm_input[0])

        eos_token_id = self.tokenizer.eos_token_id

        for (token, prob), n in zip(
            generate_step(
                prompt_tokens,
                self.llm.model,
                temp,
                repetition_penalty,
                repetition_context_size,
            ),
            range(max_new_tokens),
        ):
            # identify text to yield
            text: Optional[str] = None
            text = self.tokenizer.decode(token.item())

            # yield text, if any
            if text:
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(text, chunk=chunk)

            # break if stop sequence found
            if token == eos_token_id or (stop is not None and text in stop):
                break
