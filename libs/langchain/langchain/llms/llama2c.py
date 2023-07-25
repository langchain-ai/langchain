import logging
import os
import shlex
import subprocess
from typing import Any, Dict, List, Optional

from pydantic import root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

logger = logging.getLogger(__name__)


class Llama2c(LLM):
    """llama2c model.

    To use, you should have the llama2c inference c code complied
    (gcc -O3 -o run run.c -lm) and have a model downloaded, with
    path to the Llama2c model as a named parameter to the constructor.
    Check out: https://github.com/karpathy/llama2.c

    Example:
        .. code-block:: python
            from langchain.llms import Llama2c
            llm = Llama2c(directory="/path/to/llama2.c", 
            model_dir="rel/path/to/weights")
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

        model_param_names = [
            # List your model's parameters here...
        ]
        model_params = {k: values[k] for k in model_param_names}
        values["client"] = model_params

        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "llama2c"

    def _get_parameters(self, stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Performs sanity check, preparing parameters in format needed by llama2c.

        Args:
            stop (Optional[List[str]]): List of stop sequences for llama2c.

        Returns:
            Dictionary containing the combined parameters.
        """

        # Placeholder for future parameters
        # params = self._default_params

        return None

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Llama2c binary and return the output."""

        # params = self._get_parameters(stop)
        # params_str = " ".join(f"--{k}={v}" for k, v in params.items())
        # cmd = f"{self.inference_path} {self.model_dir} {params_str}"
        original_dir = os.getcwd()
        os.chdir(self.directory)
        inference_binary = os.path.join(self.directory, "run")
        cmd = f"{inference_binary} {self.model_dir}"
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
        """Call the Llama2c binary and stream the output."""

        # params = self._get_parameters(stop)
        # params_str = " ".join(f"--{k}={v}" for k, v in params.items())
        # cmd = f"{self.inference_path} {self.model_dir} {params_str}"
        original_dir = os.getcwd()
        os.chdir(self.directory)
        inference_binary = os.path.join(self.directory, "run")
        cmd = f"{inference_binary} {self.model_dir}"
        process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, text=True)
        os.chdir(original_dir)
        output = []

        # Read output from stdout
        output = []
        while True:
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
