"""Base interface for large language models to expose."""
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import yaml
from pydantic import BaseModel, Extra, Field, validator

import langchain
from langchain.callbacks import get_callback_manager
from langchain.callbacks.base import BaseCallbackManager
from langchain.schema import Generation, LLMResult


def _get_verbosity() -> bool:
    return langchain.verbose


class BaseLLM(BaseModel, ABC):
    """LLM wrapper should take in a prompt and return a string."""

    cache: Optional[bool] = None
    verbose: bool = Field(default_factory=_get_verbosity)
    """Whether to print out response text."""
    callback_manager: BaseCallbackManager = Field(default_factory=get_callback_manager)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @validator("callback_manager", pre=True, always=True)
    def set_callback_manager(
        cls, callback_manager: Optional[BaseCallbackManager]
    ) -> BaseCallbackManager:
        """If callback manager is None, set it.

        This allows users to pass in None as callback manager, which is a nice UX.
        """
        return callback_manager or get_callback_manager()

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
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Run the LLM on the given prompts."""

    def generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        disregard_cache = self.cache is not None and not self.cache
        if langchain.llm_cache is None or disregard_cache:
            # This happens when langchain.cache is None, but self.cache is True
            if self.cache is not None and self.cache:
                raise ValueError(
                    "Asked to cache, but no cache found at `langchain.cache`."
                )
            if self.verbose:
                self.callback_manager.on_llm_start(
                    {"name": self.__class__.__name__}, prompts
                )
            try:
                output = self._generate(prompts, stop=stop)
            except Exception as e:
                if self.verbose:
                    self.callback_manager.on_llm_error(e)
                raise e
            if self.verbose:
                self.callback_manager.on_llm_end(output)
            return output
        params = self._llm_dict()
        params["stop"] = stop
        llm_string = str(sorted([(k, v) for k, v in params.items()]))
        missing_prompts = []
        missing_prompt_idxs = []
        existing_prompts = {}
        for i, prompt in enumerate(prompts):
            cache_val = langchain.llm_cache.lookup(prompt, llm_string)
            if isinstance(cache_val, list):
                existing_prompts[i] = cache_val
            else:
                missing_prompts.append(prompt)
                missing_prompt_idxs.append(i)
        if self.verbose:
            self.callback_manager.on_llm_start(
                {"name": self.__class__.__name__}, missing_prompts
            )
        try:
            new_results = self._generate(missing_prompts, stop=stop)
        except Exception as e:
            if self.verbose:
                self.callback_manager.on_llm_error(e)
            raise e
        if self.verbose:
            self.callback_manager.on_llm_end(new_results)
        for i, result in enumerate(new_results.generations):
            existing_prompts[missing_prompt_idxs[i]] = result
            prompt = prompts[missing_prompt_idxs[i]]
            langchain.llm_cache.update(prompt, llm_string, result)
        generations = [existing_prompts[i] for i in range(len(prompts))]
        return LLMResult(generations=generations, llm_output=new_results.llm_output)

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text."""
        # TODO: this method may not be exact.
        # TODO: this method may differ based on model (eg codex).
        try:
            from transformers import GPT2TokenizerFast
        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "This is needed in order to calculate get_num_tokens. "
                "Please it install it with `pip install transformers`."
            )
        # create a GPT-3 tokenizer instance
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        # tokenize the text using the GPT-3 tokenizer
        tokenized_text = tokenizer.tokenize(text)

        # calculate the number of tokens in the tokenized text
        return len(tokenized_text)

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Check Cache and run the LLM on the given prompt and input."""
        return self.generate([prompt], stop=stop).generations[0][0].text

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

    def _llm_dict(self) -> Dict:
        """Return a dictionary of the prompt."""
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
        prompt_dict = self._llm_dict()

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
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Run the LLM on the given prompt and input."""

    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        # TODO: add caching here.
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)
