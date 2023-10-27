"""Chain that just formats a prompt and calls an LLM."""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

from langchain.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainRun,
    CallbackManager,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain.chains.base import Chain
from langchain.load.dump import dumpd
from langchain.prompts.prompt import PromptTemplate
from langchain.pydantic_v1 import Extra, Field
from langchain.schema import (
    BaseLLMOutputParser,
    BaseMessage,
    BasePromptTemplate,
    ChatGeneration,
    Generation,
    LLMResult,
    PromptValue,
    StrOutputParser,
)
from langchain.schema.language_model import (
    BaseLanguageModel,
    LanguageModelInput,
)
from langchain.schema.runnable import (
    Runnable,
    RunnableBinding,
    RunnableBranch,
    RunnableWithFallbacks,
)
from langchain.schema.runnable.configurable import DynamicRunnable
from langchain.utils.input import get_colored_text


class LLMChain(Chain):
    """Chain to run queries against LLMs.

    Example:
        .. code-block:: python

            from langchain.chains import LLMChain
            from langchain.llms import OpenAI
            from langchain.prompts import PromptTemplate
            prompt_template = "Tell me a {adjective} joke"
            prompt = PromptTemplate(
                input_variables=["adjective"], template=prompt_template
            )
            llm = LLMChain(llm=OpenAI(), prompt=prompt)
    """

    @classmethod
    def is_lc_serializable(self) -> bool:
        return True

    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: Union[
        Runnable[LanguageModelInput, str], Runnable[LanguageModelInput, BaseMessage]
    ]
    """Language model to call."""
    output_key: str = "text"  #: :meta private:
    output_parser: BaseLLMOutputParser = Field(default_factory=StrOutputParser)
    """Output parser to use.
    Defaults to one that takes the most likely string but does not change it 
    otherwise."""
    return_final_only: bool = True
    """Whether to return only the final parsed result. Defaults to True.
    If false, will return a bunch of extra information about the generation."""
    llm_kwargs: dict = Field(default_factory=dict)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        if self.return_final_only:
            return [self.output_key]
        else:
            return [self.output_key, "full_generation"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        response = self.generate([inputs], run_manager=run_manager)
        return self.create_outputs(response)[0]

    def generate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        prompts, stop = self.prep_prompts(input_list, run_manager=run_manager)
        callbacks = run_manager.get_child() if run_manager else None
        if isinstance(self.llm, BaseLanguageModel):
            return self.llm.generate_prompt(
                prompts,
                stop,
                callbacks=callbacks,
                **self.llm_kwargs,
            )
        else:
            results = self.llm.bind(stop=stop, **self.llm_kwargs).batch(
                cast(List, prompts), {"callbacks": callbacks}
            )
            generations: List[List[Generation]] = []
            for res in results:
                if isinstance(res, BaseMessage):
                    generations.append([ChatGeneration(message=res)])
                else:
                    generations.append([Generation(text=res)])
            return LLMResult(generations=generations)

    async def agenerate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        prompts, stop = await self.aprep_prompts(input_list, run_manager=run_manager)
        callbacks = run_manager.get_child() if run_manager else None
        if isinstance(self.llm, BaseLanguageModel):
            return await self.llm.agenerate_prompt(
                prompts,
                stop,
                callbacks=callbacks,
                **self.llm_kwargs,
            )
        else:
            results = await self.llm.bind(stop=stop, **self.llm_kwargs).abatch(
                cast(List, prompts), {"callbacks": callbacks}
            )
            generations: List[List[Generation]] = []
            for res in results:
                if isinstance(res, BaseMessage):
                    generations.append([ChatGeneration(message=res)])
                else:
                    generations.append([Generation(text=res)])
            return LLMResult(generations=generations)

    def prep_prompts(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Tuple[List[PromptValue], Optional[List[str]]]:
        """Prepare prompts from inputs."""
        stop = None
        if len(input_list) == 0:
            return [], stop
        if "stop" in input_list[0]:
            stop = input_list[0]["stop"]
        prompts = []
        for inputs in input_list:
            selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
            prompt = self.prompt.format_prompt(**selected_inputs)
            _colored_text = get_colored_text(prompt.to_string(), "green")
            _text = "Prompt after formatting:\n" + _colored_text
            if run_manager:
                run_manager.on_text(_text, end="\n", verbose=self.verbose)
            if "stop" in inputs and inputs["stop"] != stop:
                raise ValueError(
                    "If `stop` is present in any inputs, should be present in all."
                )
            prompts.append(prompt)
        return prompts, stop

    async def aprep_prompts(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Tuple[List[PromptValue], Optional[List[str]]]:
        """Prepare prompts from inputs."""
        stop = None
        if len(input_list) == 0:
            return [], stop
        if "stop" in input_list[0]:
            stop = input_list[0]["stop"]
        prompts = []
        for inputs in input_list:
            selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
            prompt = self.prompt.format_prompt(**selected_inputs)
            _colored_text = get_colored_text(prompt.to_string(), "green")
            _text = "Prompt after formatting:\n" + _colored_text
            if run_manager:
                await run_manager.on_text(_text, end="\n", verbose=self.verbose)
            if "stop" in inputs and inputs["stop"] != stop:
                raise ValueError(
                    "If `stop` is present in any inputs, should be present in all."
                )
            prompts.append(prompt)
        return prompts, stop

    def apply(
        self, input_list: List[Dict[str, Any]], callbacks: Callbacks = None
    ) -> List[Dict[str, str]]:
        """Utilize the LLM generate method for speed gains."""
        callback_manager = CallbackManager.configure(
            callbacks, self.callbacks, self.verbose
        )
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            {"input_list": input_list},
        )
        try:
            response = self.generate(input_list, run_manager=run_manager)
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise e
        outputs = self.create_outputs(response)
        run_manager.on_chain_end({"outputs": outputs})
        return outputs

    async def aapply(
        self, input_list: List[Dict[str, Any]], callbacks: Callbacks = None
    ) -> List[Dict[str, str]]:
        """Utilize the LLM generate method for speed gains."""
        callback_manager = AsyncCallbackManager.configure(
            callbacks, self.callbacks, self.verbose
        )
        run_manager = await callback_manager.on_chain_start(
            dumpd(self),
            {"input_list": input_list},
        )
        try:
            response = await self.agenerate(input_list, run_manager=run_manager)
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise e
        outputs = self.create_outputs(response)
        await run_manager.on_chain_end({"outputs": outputs})
        return outputs

    @property
    def _run_output_key(self) -> str:
        return self.output_key

    def create_outputs(self, llm_result: LLMResult) -> List[Dict[str, Any]]:
        """Create outputs from response."""
        result = [
            # Get the text of the top generated string.
            {
                self.output_key: self.output_parser.parse_result(generation),
                "full_generation": generation,
            }
            for generation in llm_result.generations
        ]
        if self.return_final_only:
            result = [{self.output_key: r[self.output_key]} for r in result]
        return result

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        response = await self.agenerate([inputs], run_manager=run_manager)
        return self.create_outputs(response)[0]

    def predict(self, callbacks: Callbacks = None, **kwargs: Any) -> str:
        """Format prompt with kwargs and pass to LLM.

        Args:
            callbacks: Callbacks to pass to LLMChain
            **kwargs: Keys to pass to prompt template.

        Returns:
            Completion from LLM.

        Example:
            .. code-block:: python

                completion = llm.predict(adjective="funny")
        """
        return self(kwargs, callbacks=callbacks)[self.output_key]

    async def apredict(self, callbacks: Callbacks = None, **kwargs: Any) -> str:
        """Format prompt with kwargs and pass to LLM.

        Args:
            callbacks: Callbacks to pass to LLMChain
            **kwargs: Keys to pass to prompt template.

        Returns:
            Completion from LLM.

        Example:
            .. code-block:: python

                completion = llm.predict(adjective="funny")
        """
        return (await self.acall(kwargs, callbacks=callbacks))[self.output_key]

    def predict_and_parse(
        self, callbacks: Callbacks = None, **kwargs: Any
    ) -> Union[str, List[str], Dict[str, Any]]:
        """Call predict and then parse the results."""
        warnings.warn(
            "The predict_and_parse method is deprecated, "
            "instead pass an output parser directly to LLMChain."
        )
        result = self.predict(callbacks=callbacks, **kwargs)
        if self.prompt.output_parser is not None:
            return self.prompt.output_parser.parse(result)
        else:
            return result

    async def apredict_and_parse(
        self, callbacks: Callbacks = None, **kwargs: Any
    ) -> Union[str, List[str], Dict[str, str]]:
        """Call apredict and then parse the results."""
        warnings.warn(
            "The apredict_and_parse method is deprecated, "
            "instead pass an output parser directly to LLMChain."
        )
        result = await self.apredict(callbacks=callbacks, **kwargs)
        if self.prompt.output_parser is not None:
            return self.prompt.output_parser.parse(result)
        else:
            return result

    def apply_and_parse(
        self, input_list: List[Dict[str, Any]], callbacks: Callbacks = None
    ) -> Sequence[Union[str, List[str], Dict[str, str]]]:
        """Call apply and then parse the results."""
        warnings.warn(
            "The apply_and_parse method is deprecated, "
            "instead pass an output parser directly to LLMChain."
        )
        result = self.apply(input_list, callbacks=callbacks)
        return self._parse_generation(result)

    def _parse_generation(
        self, generation: List[Dict[str, str]]
    ) -> Sequence[Union[str, List[str], Dict[str, str]]]:
        if self.prompt.output_parser is not None:
            return [
                self.prompt.output_parser.parse(res[self.output_key])
                for res in generation
            ]
        else:
            return generation

    async def aapply_and_parse(
        self, input_list: List[Dict[str, Any]], callbacks: Callbacks = None
    ) -> Sequence[Union[str, List[str], Dict[str, str]]]:
        """Call apply and then parse the results."""
        warnings.warn(
            "The aapply_and_parse method is deprecated, "
            "instead pass an output parser directly to LLMChain."
        )
        result = await self.aapply(input_list, callbacks=callbacks)
        return self._parse_generation(result)

    @property
    def _chain_type(self) -> str:
        return "llm_chain"

    @classmethod
    def from_string(cls, llm: BaseLanguageModel, template: str) -> LLMChain:
        """Create LLMChain from LLM and template."""
        prompt_template = PromptTemplate.from_template(template)
        return cls(llm=llm, prompt=prompt_template)

    def _get_num_tokens(self, text: str) -> int:
        return _get_language_model(self.llm).get_num_tokens(text)


def _get_language_model(llm_like: Runnable) -> BaseLanguageModel:
    if isinstance(llm_like, BaseLanguageModel):
        return llm_like
    elif isinstance(llm_like, RunnableBinding):
        return _get_language_model(llm_like.bound)
    elif isinstance(llm_like, RunnableWithFallbacks):
        return _get_language_model(llm_like.runnable)
    elif isinstance(llm_like, (RunnableBranch, DynamicRunnable)):
        return _get_language_model(llm_like.default)
    else:
        raise ValueError(
            f"Unable to extract BaseLanguageModel from llm_like object of type "
            f"{type(llm_like)}"
        )
