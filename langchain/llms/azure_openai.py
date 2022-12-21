"""Wrapper around Azure OpenAI APIs."""
import sys
from typing import Any, Dict, Generator, List, Mapping, Optional

from pydantic import BaseModel, Extra, Field, root_validator

from langchain.llms import OpenAI
from langchain.llms.base import BaseLLM, LLMResult
from langchain.schema import Generation
from langchain.utils import get_from_dict_or_env


class AzureOpenAI(OpenAI):
    """Wrapper around Azure OpenAI large language models.

    To use, you should have the ``openai`` python package installed, and the following
    environment variables set:

    ``OPENAI_API_KEY``: set with your API key
    ``OPENAI_API_VERSION``: set with the API version you want to use - use ``2022-12-01`` for the released version
    ``OPENAI_API_BASE``: set with the base URL for your Azure OpenAI resource
    ``OPENAI_API_TYPE``: set to 'azure'

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain import AzureOpenAI
            openai = AzureOpenAI(model_name="text-davinci-003", deployment_name="mydeployment")
    """

    deployment_name: str = ""
    """Azure OpenAI deployment name to use."""

    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Call out to OpenAI's endpoint with k unique prompts.

        Args:
            prompts: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The full LLM output.

        Example:
            .. code-block:: python

                response = openai.generate(["Tell me a joke."])
        """
        # TODO: write a unit test for this
        params = self._default_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop

        if params["max_tokens"] == -1:
            if len(prompts) != 1:
                raise ValueError(
                    "max_tokens set to -1 not supported for multiple inputs."
                )
            params["max_tokens"] = self.max_tokens_for_prompt(prompts[0])
        sub_prompts = [
            prompts[i : i + self.batch_size]
            for i in range(0, len(prompts), self.batch_size)
        ]
        choices = []
        token_usage = {}
        # Get the token usage from the response.
        # Includes prompt, completion, and total tokens used.
        _keys = ["completion_tokens", "prompt_tokens", "total_tokens"]
        for _prompts in sub_prompts:
            response = self.client.create(
                engine=self.deployment_name, prompt=_prompts, **params
            )
            choices.extend(response["choices"])
            for _key in _keys:
                if _key not in token_usage:
                    token_usage[_key] = response["usage"][_key]
                else:
                    token_usage[_key] += response["usage"][_key]
        generations = []
        for i, prompt in enumerate(prompts):
            sub_choices = choices[i * self.n : (i + 1) * self.n]
            generations.append(
                [Generation(text=choice["text"]) for choice in sub_choices]
            )
        return LLMResult(
            generations=generations, llm_output={"token_usage": token_usage}
        )

    def stream(self, prompt: str) -> Generator:
        """Call OpenAI with streaming flag and return the resulting generator.

        BETA: this is a beta feature while we figure out the right abstraction.
        Once that happens, this interface could change.

        Args:
            prompt: The prompts to pass into the model.

        Returns:
            A generator representing the stream of tokens from OpenAI.

        Example:
            .. code-block:: python

                generator = openai.stream("Tell me a joke.")
                for token in generator:
                    yield token
        """
        params = self._default_params
        if params["best_of"] != 1:
            raise ValueError("OpenAI only supports best_of == 1 for streaming")
        params["stream"] = True
        generator = self.client.create(
            engine=self.deployment_name, prompt=prompt, **params
        )

        return generator

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"model_name": self.model_name, "deployment_name": self.deployment_name},
            **self._default_params,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "azure_openai"
