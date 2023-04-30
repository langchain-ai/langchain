"""Chain that interprets a prompt and executes bash code to perform bash operations."""
import logging
import re
from typing import Any, Dict, List

from pydantic import Extra, Field

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.llm_bash.prompt import PROMPT
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseLanguageModel, BaseOutputParser, OutputParserException
from langchain.utilities.bash import BashProcess

logger = logging.getLogger(__name__)


class BashOutputParser(BaseOutputParser):
    """Parser for bash output."""

    def parse(self, text: str) -> List[str]:
        if "```bash" in text:
            return self.get_code_blocks(text)
        else:
            raise OutputParserException(
                f"Failed to parse bash output. Got: {text}",
            )

    @staticmethod
    def get_code_blocks(t: str) -> List[str]:
        """Get multiple code blocks from the LLM result."""
        code_blocks: List[str] = []
        # Bash markdown code blocks
        pattern = re.compile(r"```bash(.*?)(?:\n\s*)```", re.DOTALL)
        for match in pattern.finditer(t):
            matched = match.group(1).strip()
            if matched:
                code_blocks.extend(
                    [line for line in matched.split("\n") if line.strip()]
                )

        return code_blocks


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
    output_parser: BaseOutputParser = Field(default_factory=BashOutputParser)
    bash_process: BashProcess = Field(default_factory=BashProcess)  #: :meta private:

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

        self.callback_manager.on_text(inputs[self.input_key], verbose=self.verbose)

        t = llm_executor.predict(question=inputs[self.input_key])
        self.callback_manager.on_text(t, color="green", verbose=self.verbose)
        t = t.strip()
        try:
            command_list = self.output_parser.parse(t)
        except OutputParserException as e:
            self.callback_manager.on_chain_error(e, verbose=self.verbose)
            raise e

        if self.verbose:
            self.callback_manager.on_text("\nCode: ", verbose=self.verbose)
            self.callback_manager.on_text(
                str(command_list), color="yellow", verbose=self.verbose
            )

        output = self.bash_process.run(command_list)

        self.callback_manager.on_text("\nAnswer: ", verbose=self.verbose)
        self.callback_manager.on_text(output, color="yellow", verbose=self.verbose)
        return {self.output_key: output}

    @property
    def _chain_type(self) -> str:
        return "llm_bash_chain"

    @classmethod
    def from_bash_process(
        cls,
        bash_process: BashProcess,
        llm: BaseLanguageModel,
        **kwargs: Any,
    ) -> "LLMBashChain":
        """Create a LLMBashChain from a BashProcess."""
        return cls(llm=llm, bash_process=bash_process, **kwargs)
