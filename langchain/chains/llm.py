"""Chain that just formats a prompt and calls an LLM."""
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml
from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.input import print_text
from langchain.llms.base import LLM
from langchain.llms.loading import load_llm, load_llm_from_config
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.loading import load_prompt, load_prompt_from_config


class LLMChain(Chain, BaseModel):
    """Chain to run queries against LLMs.

    Example:
        .. code-block:: python

            from langchain import LLMChain, OpenAI, PromptTemplate
            prompt_template = "Tell me a {adjective} joke"
            prompt = PromptTemplate(
                input_variables=["adjective"], template=prompt_template
            )
            llm = LLMChain(llm=OpenAI(), prompt=prompt)
    """

    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: LLM
    """LLM wrapper to use."""
    output_key: str = "text"  #: :meta private:

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
        return [self.output_key]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
        prompt = self.prompt.format(**selected_inputs)
        if self.verbose:
            print("Prompt after formatting:")
            print_text(prompt, color="green", end="\n")
        kwargs = {}
        if "stop" in inputs:
            kwargs["stop"] = inputs["stop"]
        response = self.llm(prompt, **kwargs)
        return {self.output_key: response}

    def predict(self, **kwargs: Any) -> str:
        """Format prompt with kwargs and pass to LLM.

        Args:
            **kwargs: Keys to pass to prompt template.

        Returns:
            Completion from LLM.

        Example:
            .. code-block:: python

                completion = llm.predict(adjective="funny")
        """
        return self(kwargs)[self.output_key]

    def predict_and_parse(self, **kwargs: Any) -> Union[str, List[str], Dict[str, str]]:
        """Call predict and then parse the results."""
        result = self.predict(**kwargs)
        if self.prompt.output_parser is not None:
            return self.prompt.output_parser.parse(result)
        else:
            return result

    @classmethod
    def from_config(cls, config: dict):
        """Load LLMChain from Config."""
        if "memory" in config:
            raise ValueError("Loading memory not currently supported.")
        # Load the prompt.
        if "prompt_path" in config:
            if "prompt" in config:
                raise ValueError(
                    "Only one of prompt and prompt_path should " "be specified."
                )
            config["prompt"] = load_prompt(config.pop("prompt_path"))
        else:
            config["prompt"] = load_prompt_from_config(config["prompt"])

        # Load the LLM
        if "llm_path" in config:
            if "llm" in config:
                raise ValueError(
                    "Only one of prompt and prompt_path should " "be specified."
                )
            config["llm"] = load_llm(config.pop("llm_path"))
        else:
            config["llm"] = load_prompt_from_config(config["llm"])

        return cls(**config)

    def save(self, file_path: Union[Path, str]) -> None:
        # Convert file to Path object.
        if isinstance(file_path, str):
            save_path = Path(file_path)
        else:
            save_path = file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        info_directory = self.dict()

        if self.memory is not None:
            raise ValueError("Saving Memory not Currently Supported.")
        del info_directory["memory"]

        # Save prompts and llms separately
        del info_directory["prompt"]
        del info_directory["llm"]

        prompt_file = directory_path / "llm_prompt.yaml"
        llm_file = directory_path / "llm.yaml"

        info_directory["prompt_path"] = str(prompt_file)
        info_directory["llm_path"] = str(llm_file)
        info_directory["_type"] = "llm"

        # Save prompt and llm associated with LLM Chain
        self.prompt.save(prompt_file)
        self.llm.save(llm_file)

        with open(save_path, "w") as f:
            yaml.dump(info_directory, f, default_flow_style=False)
