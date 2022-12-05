import subprocess
from typing import Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain

class BashChain(Chain, BaseModel):
    """Chain to execute Bash code.
    Example:
        .. code-block:: python
            from langchain import BashChain, OpenAI
            bash = BashChain()
    """

    input_key: str = "command"  #: :meta private:
    output_key: str = "output"  #: :meta private:

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
        command = inputs[self.input_key]
        try:
            output = subprocess.check_output(command, shell=True).decode()
        except subprocess.CalledProcessError as error:
            output = str(error)

        return {self.output_key: output}
