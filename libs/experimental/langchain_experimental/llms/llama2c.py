import logging
import os
import shlex
import subprocess
from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from pydantic import root_validator

logger = logging.getLogger(__name__)


class Llama2c(LLM):
    """llama2c model.

    Clone:
    git clone https://github.com/karpathy/llama2.c

    Get the inference inference c code w/ prompt:
     https://github.com/karpathy/llama2.c/issues/62#issue-1819724122

    Compile it:
    gcc -O3 -o run run_with_prompt.c -lm -funsafe-math-optimizations
      -Ofast -ffast-mat
    .
    Example:
        .. code-block:: python
            from langchain_experimental.llms.llama2c import Llama2c
            llm = Llama2c(directory="/Users/rlm/Desktop/Code/llama2.c/",
              model_dir="out44m/model44m.bin")
    """

    directory: str  # The path to the Llama2c directory.
    model_dir: str  # Relative path where the model weights are stored.

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate the paths to the binary and weights directory."""
        inference_path = values["directory"]

        # Check if the binary and directory exist
        if not os.path.exists(inference_path):
            raise ValueError(f"Inference binary not found: {inference_path}")

        model_param_names: List[str] = [
            # List your model's parameters here...
        ]
        model_params = {k: values[k] for k in model_param_names}
        values["client"] = model_params

        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "llama2c"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Llama2c binary and return the output."""

        original_dir = os.getcwd()
        os.chdir(self.directory)
        inference_binary = os.path.join(self.directory, "run")
        # See: https://github.com/karpathy/llama2.c/issues/62#issue-1819724122
        cmd = f"{inference_binary} {self.model_dir} 0.0 256 {shlex.quote(prompt)}"
        process = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
        os.chdir(original_dir)
        if process.returncode != 0:
            raise RuntimeError(f"Model execution failed with error: {process.stderr}")
        return process.stdout

    def stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Llama2c binary and stream output."""

        original_dir = os.getcwd()
        os.chdir(self.directory)
        inference_binary = os.path.join(self.directory, "run")
        # See: https://github.com/karpathy/llama2.c/issues/62#issue-1819724122
        cmd = f"{inference_binary} {self.model_dir} 0.0 256 {shlex.quote(prompt)}"
        process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, text=True)
        os.chdir(original_dir)

        # Read output from stdout
        output: List[str] = []
        while True:
            if process.stdout is not None:
                char = process.stdout.read(1)
                if char == "" and process.poll() is not None:
                    break
                if char:
                    output.append(char)
                    print(char, end="")  # print character in 'real-time'

        # Wait for the subprocess to finish
        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"Model execution failed with error: {process.stderr}")

        return "".join(output)
