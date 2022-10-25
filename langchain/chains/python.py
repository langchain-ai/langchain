"""Chain that runs python code.

Heavily borrowed from https://replit.com/@amasad/gptpy?v=1#main.py
"""
import sys
from io import StringIO
from typing import Dict, List

from pydantic import BaseModel

from langchain.chains.base import Chain


class PythonChain(Chain, BaseModel):
    """Chain to run python code."""

    input_key: str = "code"  #: :meta private:
    output_key: str = "output"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return output key.

        :meta private:
        """
        return [self.output_key]

    def _run(self, inputs: Dict[str, str]) -> Dict[str, str]:
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        exec(inputs[self.input_key])
        sys.stdout = old_stdout
        output = mystdout.getvalue()
        return {self.output_key: output}

    def run(self, code: str) -> str:
        """More user-friendly interface for interfacing with python."""
        return self({self.input_key: code})[self.output_key]
