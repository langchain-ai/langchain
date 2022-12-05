import subprocess
from typing import Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain

class BashChain(Chain, BaseModel):
    """Chain to execute a list of Bash code.
    Example:
        .. code-block:: python
            from langchain import BashChain, OpenAI
            bash = BashChain()
    """

    input_key: str = "commands"  #: :meta private:
    output_key: str = "outputs"  #: :meta private:

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

    def _call(self, inputs: Dict[str, List[str]]) -> Dict[str, List[str]]:
        commands = inputs[self.input_key]
        outputs = []
        for command in commands:
            try:
                output = subprocess.check_output(command, shell=True).decode()
                outputs.append(output)
            except subprocess.CalledProcessError as error:
                outputs.append(str(error))
                return {self.output_key: outputs}

        return {self.output_key: outputs}
