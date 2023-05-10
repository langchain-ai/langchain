"""Base interface for large language models to expose."""
import inspect
import json
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import yaml
from pydantic import Extra, Field, root_validator, validator

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
from langchain.schema import Generation, LLMResult, PromptValue


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
                {"name": self.__class__.__name__}, prompts
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
        params = self.dict()
        params["stop"] = stop
        (
            existing_prompts,
            llm_string,
            missing_prompt_idxs,
            missing_prompts,
        ) = get_prompts(params, prompts)
        if len(missing_prompts) > 0:
            run_manager = callback_manager.on_llm_start(
                {"name": self.__class__.__name__}, missing_prompts
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
                {"name": self.__class__.__name__}, prompts
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
        params = self.dict()
        params["stop"] = stop
        (
            existing_prompts,
            llm_string,
            missing_prompt_idxs,
            missing_prompts,
        ) = get_prompts(params, prompts)
        if len(missing_prompts) > 0:
            run_manager = await callback_manager.on_llm_start(
                {"name": self.__class__.__name__},
                missing_prompts,
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
