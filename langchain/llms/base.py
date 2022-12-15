"""Base interface for large language models to expose."""
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Union

import yaml
from pydantic import BaseModel, Extra

import langchain
from langchain.schema import Generation


class LLMResult(NamedTuple):
    """Class that contains all relevant information for an LLM Result."""

    generations: List[List[Generation]]
    """List of the things generated. This is List[List[]] because
    each input could have multiple generations."""
    llm_output: Optional[dict] = None
    """For arbitrary LLM provider specific output."""


class LLM(BaseModel, ABC):
    """LLM wrapper should take in a prompt and return a string."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        # TODO: add caching here.
        generations = []
        for prompt in prompts:
            text = self(prompt, stop=stop)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    def generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        if langchain.llm_cache is None:
            return self._generate(prompts, stop=stop)
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
        new_results = self._generate(missing_prompts, stop=stop)
        for i, result in enumerate(new_results.generations):
            existing_prompts[i] = result
            prompt = prompts[i]
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
                "This is needed in order to calculate max_tokens_for_prompt. "
                "Please it install it with `pip install transformers`."
            )
        # create a GPT-3 tokenizer instance
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        # tokenize the text using the GPT-3 tokenizer
        tokenized_text = tokenizer.tokenize(text)

        # calculate the number of tokens in the tokenized text
        return len(tokenized_text)

    @abstractmethod
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Run the LLM on the given prompt and input."""

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Check Cache and run the LLM on the given prompt and input."""
        if langchain.llm_cache is None:
            return self._call(prompt, stop=stop)
        params = self._llm_dict()
        params["stop"] = stop
        llm_string = str(sorted([(k, v) for k, v in params.items()]))
        if langchain.cache is not None:
            cache_val = langchain.llm_cache.lookup(prompt, llm_string)
            if cache_val is not None:
                if isinstance(cache_val, str):
                    return cache_val
                else:
                    return cache_val[0].text
        return_val = self._call(prompt, stop=stop)
        if langchain.cache is not None:
            langchain.llm_cache.update(prompt, llm_string, return_val)
        return return_val

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
