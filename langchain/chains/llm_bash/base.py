"""Chain that interprets a prompt and executes bash code to perform bash operations."""
from typing import Dict, List

from pydantic import Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.llm_bash.prompt import PROMPT
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseLanguageModel
from langchain.utilities.bash import BashProcess


class LLMBashChain(Chain):
    """Chain that interprets a prompt and executes bash code to perform bash operations.

    Example:
        .. code-block:: python

            from langchain import LLMBashChain, OpenAI
            llm_bash = LLMBashChain(llm=OpenAI())
    """

    llm: BaseLanguageModel
    """LLM wrapper to use."""
    input_key: str = "question"  #: :meta private:
    output_key: str = "answer"  #: :meta private:
    prompt: BasePromptTemplate = PROMPT
    persistent: bool = False

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        llm_executor = LLMChain(prompt=self.prompt, llm=self.llm)
        bash_executor = BashProcess(persistent=self.persistent)
        self.callback_manager.on_text(inputs[self.input_key], verbose=self.verbose)

        t = llm_executor.predict(question=inputs[self.input_key])
        self.callback_manager.on_text(t, color="green", verbose=self.verbose)

        t = t.strip()
        if "```bash" in t:
            # Split the string into a list of substrings
            command_list = self.get_code(t)
            command_list = [s for s in command_list if s]

            if self.verbose:
                self.callback_manager.on_text("\nCode: ", verbose=self.verbose)
                self.callback_manager.on_text(command_list, color="yellow", verbose=self.verbose)

            output = bash_executor.run(command_list)

            self.callback_manager.on_text("\nAnswer: ", verbose=self.verbose)
            self.callback_manager.on_text(output, color="yellow", verbose=self.verbose)
        else:
            output = t
            # raise ValueError(f"unknown format from LLM: {t}")
        return {self.output_key: output}

    def get_code_and_remainder(self, t: str) -> str:
        """Get a python code block from the LLM result.

        :meta private:
        """
        splits = t.split("```bash")  
        split = splits[1].split("```")
    
        if len(splits)==2:
            return split[0], "```".join(split[1:])
        else:
            return split[0], "```".join(split[1:]) + "```bash" + "```bash".join(splits[2:])

    def get_code_blocks(self, t: str) -> List[str]:
        """Get multiple python code blocks from the LLM result.

        :meta private:
        """
        code_blocks = []
        while "```bash" in t:
            code, t = self.get_code_and_remainder(t)
            code_blocks.append(code)
        return code_blocks

    def get_code(self, t: str) -> str:
        """Get a python code block from the LLM result.

        :meta private:
        """
        return ("\n".join(self.get_code_blocks(t))).split("\n")

    @property
    def _chain_type(self) -> str:
        return "llm_bash_chain"
