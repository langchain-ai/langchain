"""Base interface for large language models to expose."""
import asyncio
import inspect
import json
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import yaml
from pydantic import BaseModel, Extra, Field, root_validator, validator

import langchain
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    CallbackManager,
    CallbackManagerForLLMRun,
    Callbacks,
)
from langchain.chains.llm import LLMChain
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import (
    AIMessage,
    BaseMessage,
    Generation,
    LLMResult,
    PromptValue,
    get_buffer_string,
)


def _get_verbosity() -> bool:
    return langchain.verbose


def get_prompts(
    params: Dict[str, Any], prompts: List[str]
) -> Tuple[Dict[int, List], str, List[int], List[str]]:
    """Get prompts that are already cached."""
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    missing_prompts = []
    missing_prompt_idxs = []
    existing_prompts = {}
    for i, prompt in enumerate(prompts):
        if langchain.llm_cache is not None:
            cache_val = langchain.llm_cache.lookup(prompt, llm_string)
            if isinstance(cache_val, list):
                existing_prompts[i] = cache_val
            else:
                missing_prompts.append(prompt)
                missing_prompt_idxs.append(i)
    return existing_prompts, llm_string, missing_prompt_idxs, missing_prompts


def update_cache(
    existing_prompts: Dict[int, List],
    llm_string: str,
    missing_prompt_idxs: List[int],
    new_results: LLMResult,
    prompts: List[str],
) -> Optional[dict]:
    """Update the cache and get the LLM output."""
    for i, result in enumerate(new_results.generations):
        existing_prompts[missing_prompt_idxs[i]] = result
        prompt = prompts[missing_prompt_idxs[i]]
        if langchain.llm_cache is not None:
            langchain.llm_cache.update(prompt, llm_string, result)
    llm_output = new_results.llm_output
    return llm_output


class BaseLLM(BaseLanguageModel, ABC):
    """LLM wrapper should take in a prompt and return a string."""

    cache: Optional[bool] = None
    verbose: bool = Field(default_factory=_get_verbosity)
    """Whether to print out response text."""
    callbacks: Callbacks = Field(default=None, exclude=True)
    callback_manager: Optional[BaseCallbackManager] = Field(default=None, exclude=True)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator()
    def raise_deprecation(cls, values: Dict) -> Dict:
        """Raise deprecation warning if callback_manager is used."""
        if values.get("callback_manager") is not None:
            warnings.warn(
                "callback_manager is deprecated. Please use callbacks instead.",
                DeprecationWarning,
            )
            values["callbacks"] = values.pop("callback_manager", None)
        return values

    @validator("verbose", pre=True, always=True)
    def set_verbose(cls, verbose: Optional[bool]) -> bool:
        """If verbose is None, set it.

        This allows users to pass in None as verbose to access the global setting.
        """
        if verbose is None:
            return _get_verbosity()
        else:
            return verbose

    @abstractmethod
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> LLMResult:
        """Run the LLM on the given prompts."""

    @abstractmethod
    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> LLMResult:
        """Run the LLM on the given prompts."""

    def generate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        prompt_strings = [p.to_string() for p in prompts]
        return self.generate(prompt_strings, stop=stop, callbacks=callbacks)

    async def agenerate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        prompt_strings = [p.to_string() for p in prompts]
        return await self.agenerate(prompt_strings, stop=stop, callbacks=callbacks)

    def generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        # If string is passed in directly no errors will be raised but outputs will
        # not make sense.
        if not isinstance(prompts, list):
            raise ValueError(
                "Argument 'prompts' is expected to be of type List[str], received"
                f" argument of type {type(prompts)}."
            )
        params = self.dict()
        params["stop"] = stop
        (
            existing_prompts,
            llm_string,
            missing_prompt_idxs,
            missing_prompts,
        ) = get_prompts(params, prompts)
        disregard_cache = self.cache is not None and not self.cache
        callback_manager = CallbackManager.configure(
            callbacks, self.callbacks, self.verbose
        )
        new_arg_supported = inspect.signature(self._generate).parameters.get(
            "run_manager"
        )
        if langchain.llm_cache is None or disregard_cache:
            # This happens when langchain.cache is None, but self.cache is True
            if self.cache is not None and self.cache:
                raise ValueError(
                    "Asked to cache, but no cache found at `langchain.cache`."
                )
            run_manager = callback_manager.on_llm_start(
                {"name": self.__class__.__name__}, prompts, invocation_params=params
            )
            try:
                output = (
                    self._generate(prompts, stop=stop, run_manager=run_manager)
                    if new_arg_supported
                    else self._generate(prompts, stop=stop)
                )
            except (KeyboardInterrupt, Exception) as e:
                run_manager.on_llm_error(e)
                raise e
            run_manager.on_llm_end(output)
            return output
        if len(missing_prompts) > 0:
            run_manager = callback_manager.on_llm_start(
                {"name": self.__class__.__name__},
                missing_prompts,
                invocation_params=params,
            )
            try:
                new_results = (
                    self._generate(missing_prompts, stop=stop, run_manager=run_manager)
                    if new_arg_supported
                    else self._generate(missing_prompts, stop=stop)
                )
            except (KeyboardInterrupt, Exception) as e:
                run_manager.on_llm_error(e)
                raise e
            run_manager.on_llm_end(new_results)
            llm_output = update_cache(
                existing_prompts, llm_string, missing_prompt_idxs, new_results, prompts
            )
        else:
            llm_output = {}
        generations = [existing_prompts[i] for i in range(len(prompts))]
        return LLMResult(generations=generations, llm_output=llm_output)

    async def agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        params = self.dict()
        params["stop"] = stop
        (
            existing_prompts,
            llm_string,
            missing_prompt_idxs,
            missing_prompts,
        ) = get_prompts(params, prompts)
        disregard_cache = self.cache is not None and not self.cache
        callback_manager = AsyncCallbackManager.configure(
            callbacks, self.callbacks, self.verbose
        )
        new_arg_supported = inspect.signature(self._agenerate).parameters.get(
            "run_manager"
        )
        if langchain.llm_cache is None or disregard_cache:
            # This happens when langchain.cache is None, but self.cache is True
            if self.cache is not None and self.cache:
                raise ValueError(
                    "Asked to cache, but no cache found at `langchain.cache`."
                )
            run_manager = await callback_manager.on_llm_start(
                {"name": self.__class__.__name__}, prompts, invocation_params=params
            )
            try:
                output = (
                    await self._agenerate(prompts, stop=stop, run_manager=run_manager)
                    if new_arg_supported
                    else await self._agenerate(prompts, stop=stop)
                )
            except (KeyboardInterrupt, Exception) as e:
                await run_manager.on_llm_error(e, verbose=self.verbose)
                raise e
            await run_manager.on_llm_end(output, verbose=self.verbose)
            return output
        if len(missing_prompts) > 0:
            run_manager = await callback_manager.on_llm_start(
                {"name": self.__class__.__name__},
                missing_prompts,
                invocation_params=params,
            )
            try:
                new_results = (
                    await self._agenerate(
                        missing_prompts, stop=stop, run_manager=run_manager
                    )
                    if new_arg_supported
                    else await self._agenerate(missing_prompts, stop=stop)
                )
            except (KeyboardInterrupt, Exception) as e:
                await run_manager.on_llm_error(e)
                raise e
            await run_manager.on_llm_end(new_results)
            llm_output = update_cache(
                existing_prompts, llm_string, missing_prompt_idxs, new_results, prompts
            )
        else:
            llm_output = {}
        generations = [existing_prompts[i] for i in range(len(prompts))]
        return LLMResult(generations=generations, llm_output=llm_output)

    def __call__(
        self, prompt: str, stop: Optional[List[str]] = None, callbacks: Callbacks = None
    ) -> str:
        """Check Cache and run the LLM on the given prompt and input."""
        return (
            self.generate([prompt], stop=stop, callbacks=callbacks)
            .generations[0][0]
            .text
        )

    def predict(self, text: str, *, stop: Optional[Sequence[str]] = None) -> str:
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        return self(text, stop=_stop)

    def predict_messages(
        self, messages: List[BaseMessage], *, stop: Optional[Sequence[str]] = None
    ) -> BaseMessage:
        text = get_buffer_string(messages)
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        content = self(text, stop=_stop)
        return AIMessage(content=content)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    def __str__(self) -> str:
        """Get a string representation of the object for printing."""
        cls_name = f"\033[1m{self.__class__.__name__}\033[0m"
        return f"{cls_name}\nParams: {self._identifying_params}"

    @property
    @abstractmethod
    def _llm_type(self) -> str:
        """Return type of llm."""

    def dict(self, **kwargs: Any) -> Dict:
        """Return a dictionary of the LLM."""
        starter_dict = dict(self._identifying_params)
        starter_dict["_type"] = self._llm_type
        return starter_dict

    def save(self, file_path: Union[Path, str]) -> None:
        """Save the LLM.

        Args:
            file_path: Path to file to save the LLM to.

        Example:
        .. code-block:: python

            llm.save(file_path="path/llm.yaml")
        """
        # Convert file to Path object.
        if isinstance(file_path, str):
            save_path = Path(file_path)
        else:
            save_path = file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        # Fetch dictionary to save
        prompt_dict = self.dict()

        if save_path.suffix == ".json":
            with open(file_path, "w") as f:
                json.dump(prompt_dict, f, indent=4)
        elif save_path.suffix == ".yaml":
            with open(file_path, "w") as f:
                yaml.dump(prompt_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"{save_path} must be json or yaml")


class LLM(BaseLLM):
    """LLM class that expect subclasses to implement a simpler call method.

    The purpose of this class is to expose a simpler interface for working
    with LLMs, rather than expect the user to implement the full _generate method.
    """

    @abstractmethod
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """Run the LLM on the given prompt and input."""

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> str:
        """Run the LLM on the given prompt and input."""
        raise NotImplementedError("Async generation not implemented for this LLM.")

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        # TODO: add caching here.
        generations = []
        new_arg_supported = inspect.signature(self._call).parameters.get("run_manager")
        for prompt in prompts:
            text = (
                self._call(prompt, stop=stop, run_manager=run_manager)
                if new_arg_supported
                else self._call(prompt, stop=stop)
            )
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        generations = []
        new_arg_supported = inspect.signature(self._acall).parameters.get("run_manager")
        for prompt in prompts:
            text = (
                await self._acall(prompt, stop=stop, run_manager=run_manager)
                if new_arg_supported
                else await self._acall(prompt, stop=stop)
            )
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)


class BaseSmartLLM(BaseLLM):
    """
    Generalized implementation of SmartGPT (origin: https://youtu.be/wVzuvf9D9BU)

    SmartLLM passes consist of 3 steps:
    1. Ideate: Pass the user prompt to an ideation LLM n_ideas times,
       each result is an "idea"
    2. Critique: Pass the ideas to a critque LLM which looks for flaws in the ideas
       & picks the best one
    3. Resolve: Pass the critique to a resolver LLM which improves upon the best idea
       & outputs only the (improved version of) the best output

    In total, a SmartLLM pass will use n_ideas+2 LLM calls

    Note that SmartLLM will only improve results, when the underlying models have the
    capability for reflection, which smaller models often don't.

    Finally, SmartLLM assumes that each underlying LLM outputs exactly 1 result.
    """

    ideation_llm: Optional[BaseLLM] = None
    """LLM to use in ideation step. If None given, 'llm' will be used."""
    critique_llm: Optional[BaseLLM] = None
    """LLM to use in critique step. If None given, 'llm' will be used."""
    resolver_llm: Optional[BaseLLM] = None
    """LLM to use in resolve step. If None given, 'llm' will be used."""
    llm: Optional[BaseLLM] = None
    """LLM to use for each steps, if no specific llm for that step is given. """
    n_ideas: int = 3
    """Number of ideas to generate in idea step"""
    history: str = ""

    class Config:
        extra = Extra.forbid

    @root_validator
    @classmethod
    def validate_inputs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure we have an LLM for each step."""
        llm = values.get("llm")
        ideation_llm = values.get("ideation_llm")
        critique_llm = values.get("critique_llm")
        resolver_llm = values.get("resolver_llm")

        if not llm and not ideation_llm:
            raise ValueError(
                "Either ideation_llm or llm needs to be given. Pass llm, "
                "if you want to use the same llm for all steps, or pass "
                "ideation_llm, critique_llm and resolver_llm if you want "
                "to use different llms for each step."
            )
        if not llm and not critique_llm:
            raise ValueError(
                "Either critique_llm or llm needs to be given. Pass llm, "
                "if you want to use the same llm for all steps, or pass "
                "ideation_llm, critique_llm and resolver_llm if you want "
                "to use different llms for each step."
            )
        if not llm and not resolver_llm:
            raise ValueError(
                "Either resolve_llm or llm needs to be given. Pass llm, "
                "if you want to use the same llm for all steps, or pass "
                "ideation_llm, critique_llm and resolver_llm if you want "
                "to use different llms for each step."
            )
        if llm and ideation_llm and critique_llm and resolver_llm:
            raise ValueError(
                "LLMs are given for each step (ideation_llm, critique_llm,"
                " resolver_llm), but backup LLM (llm) is also given, which"
                " would not be used."
            )
        return values

    @classmethod
    @abstractmethod
    def ideation_prompt(cls) -> BasePromptTemplate:
        """Prompt used in ideation step."""

    @classmethod
    @abstractmethod
    def critique_prompt(cls) -> BasePromptTemplate:
        """Prompt used in critique step."""

    @classmethod
    @abstractmethod
    def resolve_prompt(cls) -> BasePromptTemplate:
        """Prompt used in resolve step."""

    @abstractmethod
    def _update_history_after_ideation(self, question: str, ideas: List[str]) -> None:
        """
        SmartLLM is orginially a chat-based method, so we construct a 'chat history'.
        This function return the 'chat history' after completion of the ideation step.
        """

    @abstractmethod
    def _update_history_after_critique(self, critique: str) -> None:
        """
        SmartLLM is orginially a chat-based method, so we construct a 'chat history'.
        This function return the 'chat history' after completion of the critique step.
        """

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> LLMResult:
        """..."""
        callbacks = run_manager.handlers if run_manager else None
        generations = []
        for prompt in prompts:
            ideas = self._ideate(prompt, stop, callbacks)
            self._update_history_after_ideation(prompt, ideas)
            critique = self._critique(stop, callbacks)
            self._update_history_after_critique(critique)
            resolution = self._resolve(stop, callbacks)
            generations.append(Generation(text=resolution))
        return LLMResult(generations=[generations])

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> LLMResult:
        """..."""
        callbacks = run_manager.handlers if run_manager else None
        generations = []
        for prompt in prompts:
            ideas = await self._aideate(prompt, stop, callbacks)
            self._update_history_after_ideation(prompt, ideas)
            critique = await self._acritique(stop, callbacks)
            self._update_history_after_critique(critique)
            resolution = await self._aresolve(stop, callbacks)
            generations.append(Generation(text=resolution))
        return LLMResult(generations=[generations])

    def _get_text_from_llm_result(self, result: LLMResult, step: str) -> str:
        """Between steps, only the LLM result text is passed, not the LLMResult object.
        This function extracts the text from an LLMResult."""
        if len(result.generations) != 1:
            raise ValueError(
                f"In SmartLLM the LLM result in step {step} is not "
                "exaclty 1 element. This should never happen"
            )
        if len(result.generations[0]) != 1:
            raise ValueError(
                f"In SmartLLM the LLM in step {step} returned more than "
                "1 output. SmartLLM only works with LLMs returning "
                "exactly 1 output."
            )
        return result.generations[0][0].text

    def _ideate(
        self, prompt: str, stop: Optional[List[str]] = None, callbacks: Callbacks = None
    ) -> List[str]:
        """Generate n_ideas ideas as response to user prompt."""
        llm = self.ideation_llm if self.ideation_llm else self.llm
        prompt = self.ideation_prompt().format(question=prompt)

        if llm:
            return [
                self._get_text_from_llm_result(
                    llm.generate([prompt], stop, callbacks), step="ideate"
                )
                for _ in range(self.n_ideas)
            ]
        else:
            raise ValueError("llm is none, which should never happen")

    def _critique(
        self, stop: Optional[List[str]] = None, callbacks: Callbacks = None
    ) -> str:
        """Critique each of the ideas from ideation stage & select best one."""
        llm = self.critique_llm if self.critique_llm else self.llm
        prompt = self.critique_prompt().format(
            history=self.history, n_ideas=self.n_ideas
        )
        if llm:
            return self._get_text_from_llm_result(
                llm.generate([prompt], stop, callbacks), step="critique"
            )
        else:
            raise ValueError("llm is none, which should never happen")

    def _resolve(
        self, stop: Optional[List[str]] = None, callbacks: Callbacks = None
    ) -> str:
        """Improve upon the best idea as chosen in critique step & return it."""
        llm = self.resolver_llm if self.resolver_llm else self.llm
        prompt = self.resolve_prompt().format(
            history=self.history, n_ideas=self.n_ideas
        )
        if llm:
            return self._get_text_from_llm_result(
                llm.generate([prompt], stop, callbacks), step="resolve"
            )
        else:
            raise ValueError("llm is none, which should never happen")

    async def _aideate(
        self, prompt: str, stop: Optional[List[str]] = None, callbacks: Callbacks = None
    ) -> List[str]:
        """Generate n_ideas ideas as response to user prompt."""
        llm = self.ideation_llm if self.ideation_llm else self.llm
        prompt = self.ideation_prompt().format(question=prompt)

        if llm:
            return [
                self._get_text_from_llm_result(
                    await llm.agenerate([prompt], stop, callbacks), step="ideate"
                )
                for _ in range(self.n_ideas)
            ]
        else:
            raise ValueError("llm is none, which should never happen")

    async def _acritique(
        self, stop: Optional[List[str]] = None, callbacks: Callbacks = None
    ) -> str:
        """Critique each of the ideas from ideation stage & select best one."""
        llm = self.critique_llm if self.critique_llm else self.llm
        prompt = self.critique_prompt().format(
            history=self.history, n_ideas=self.n_ideas
        )
        if llm:
            return self._get_text_from_llm_result(
                await llm.agenerate([prompt], stop, callbacks), step="critique"
            )
        else:
            raise ValueError("llm is none, which should never happen")

    async def _aresolve(
        self, stop: Optional[List[str]] = None, callbacks: Callbacks = None
    ) -> str:
        """Improve upon the best idea as chosen in critique step & return it."""
        llm = self.resolver_llm if self.resolver_llm else self.llm
        prompt = self.resolve_prompt().format(
            history=self.history, n_ideas=self.n_ideas
        )
        if llm:
            return self._get_text_from_llm_result(
                await llm.agenerate([prompt], stop, callbacks), step="resolve"
            )
        else:
            raise ValueError("llm is none, which should never happen")

    @property
    def _llm_type(self) -> str:
        if not any([self.ideation_llm, self.critique_llm, self.resolver_llm]):
            # We're using a single LLM for all steps
            if self.llm:
                return f"SmartLLM of {self.llm._llm_type}"
            else:
                raise ValueError(
                    "llm, ideation_llm, critique_llm and resolver_llm are "
                    "all none, which should never happen"
                )
        else:
            # We're using separate LLMs for all steps
            ideation_llm = self.ideation_llm if self.ideation_llm else self.llm
            critique_llm = self.critique_llm if self.critique_llm else self.llm
            resolver_llm = self.resolver_llm if self.resolver_llm else self.llm

            if ideation_llm and critique_llm and resolver_llm:
                return (
                    f"SmartLLM of {ideation_llm._llm_type}, "
                    f"{critique_llm._llm_type}, "
                    f"{resolver_llm._llm_type}, "
                )
            else:
                raise ValueError(
                    "llm, ideation_llm, critique_llm and resolver_llm are "
                    "all none, which should never happen"
                )


class SmartLLM(BaseSmartLLM):
    """Implementation of SmartGPT (origin: https://youtu.be/wVzuvf9D9BU)"""

    @classmethod
    def ideation_prompt(cls) -> BasePromptTemplate:
        return PromptTemplate(
            input_variables=["question"],
            template=(
                "Question: {question}\nAnswer: Let's work this out in a step by "
                "step way to be sure we have the right answer:"
            ),
        )

    @classmethod
    def critique_prompt(cls) -> BasePromptTemplate:
        return PromptTemplate(
            input_variables=["history", "n_ideas"],
            template=(
                "{history} You are a researcher tasked with investigating the "
                "{n_ideas} response options provided. List the flaws and faulty "
                "logic of each answer options. Let'w work this out in a step by "
                "step way to be sure we have all the errors:"
            ),
        )

    @classmethod
    def resolve_prompt(cls) -> BasePromptTemplate:
        return PromptTemplate(
            input_variables=["history", "n_ideas"],
            template=(
                "{history} You are a resolved tasked with 1) finding which of "
                "the {n_ideas} anwer options the researcher thought was best, "
                "2) improving that answer and 3) printing the answer in full. "
                "Let's work this out in a step by step way to be sure we have "
                "the right answer:"
            ),
        )

    def _update_history_after_ideation(self, question: str, ideas: List[str]) -> None:
        self.history = self.ideation_prompt().format(question=question) + "\n"
        self.history += "".join(
            [f"Response {i+1}:{idea}\n" for i, idea in enumerate(ideas)]
        )

    def _update_history_after_critique(self, critique: str) -> None:
        self.history = (
            self.critique_prompt().format(history=self.history, n_ideas=self.n_ideas)
            + "\n"
        )
        self.history += critique + "\n"
