import base64
import json
import logging
import subprocess
import textwrap
import time
from typing import Any, Dict, List, Mapping, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env, pre_init

logger = logging.getLogger(__name__)

DEFAULT_NUM_TRIES = 10
DEFAULT_SLEEP_TIME = 4


class Beam(LLM):
    """Beam API for gpt2 large language model.

    To use, you should have the ``beam-sdk`` python package installed,
    and the environment variable ``BEAM_CLIENT_ID`` set with your client id
    and ``BEAM_CLIENT_SECRET`` set with your client secret. Information on how
    to get this is available here: https://docs.beam.cloud/account/api-keys.

    The wrapper can then be called as follows, where the name, cpu, memory, gpu,
    python version, and python packages can be updated accordingly. Once deployed,
    the instance can be called.

    Example:
        .. code-block:: python

            llm = Beam(model_name="gpt2",
                name="langchain-gpt2",
                cpu=8,
                memory="32Gi",
                gpu="A10G",
                python_version="python3.8",
                python_packages=[
                    "diffusers[torch]>=0.10",
                    "transformers",
                    "torch",
                    "pillow",
                    "accelerate",
                    "safetensors",
                    "xformers",],
                max_length=50)
            llm._deploy()
            call_result = llm._call(input)

    """

    model_name: str = ""
    name: str = ""
    cpu: str = ""
    memory: str = ""
    gpu: str = ""
    python_version: str = ""
    python_packages: List[str] = []
    max_length: str = ""
    url: str = ""
    """model endpoint to use"""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not
    explicitly specified."""

    beam_client_id: str = ""
    beam_client_secret: str = ""
    app_id: Optional[str] = None

    class Config:
        """Configuration for this pydantic config."""

        extra = Extra.forbid

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in cls.__fields__.values()}

        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name in extra:
                    raise ValueError(f"Found {field_name} supplied twice.")
                logger.warning(
                    f"""{field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        beam_client_id = get_from_dict_or_env(
            values, "beam_client_id", "BEAM_CLIENT_ID"
        )
        beam_client_secret = get_from_dict_or_env(
            values, "beam_client_secret", "BEAM_CLIENT_SECRET"
        )
        values["beam_client_id"] = beam_client_id
        values["beam_client_secret"] = beam_client_secret
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "name": self.name,
            "cpu": self.cpu,
            "memory": self.memory,
            "gpu": self.gpu,
            "python_version": self.python_version,
            "python_packages": self.python_packages,
            "max_length": self.max_length,
            "model_kwargs": self.model_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "beam"

    def app_creation(self) -> None:
        """Creates a Python file which will contain your Beam app definition."""
        script = textwrap.dedent(
            """\
        import beam

        # The environment your code will run on
        app = beam.App(
            name="{name}",
            cpu={cpu},
            memory="{memory}",
            gpu="{gpu}",
            python_version="{python_version}",
            python_packages={python_packages},
        )

        app.Trigger.RestAPI(
            inputs={{"prompt": beam.Types.String(), "max_length": beam.Types.String()}},
            outputs={{"text": beam.Types.String()}},
            handler="run.py:beam_langchain",
        )

        """
        )

        script_name = "app.py"
        with open(script_name, "w") as file:
            file.write(
                script.format(
                    name=self.name,
                    cpu=self.cpu,
                    memory=self.memory,
                    gpu=self.gpu,
                    python_version=self.python_version,
                    python_packages=self.python_packages,
                )
            )

    def run_creation(self) -> None:
        """Creates a Python file which will be deployed on beam."""
        script = textwrap.dedent(
            """
        import os
        import transformers
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        model_name = "{model_name}"

        def beam_langchain(**inputs):
            prompt = inputs["prompt"]
            length = inputs["max_length"]

            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)
            encodedPrompt = tokenizer.encode(prompt, return_tensors='pt')
            outputs = model.generate(encodedPrompt, max_length=int(length),
              do_sample=True, pad_token_id=tokenizer.eos_token_id)
            output = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(output)  # noqa: T201
            return {{"text": output}}

        """
        )

        script_name = "run.py"
        with open(script_name, "w") as file:
            file.write(script.format(model_name=self.model_name))

    def _deploy(self) -> str:
        """Call to Beam."""
        try:
            import beam  # type: ignore

            if beam.__path__ == "":
                raise ImportError
        except ImportError:
            raise ImportError(
                "Could not import beam python package. "
                "Please install it with `curl "
                "https://raw.githubusercontent.com/slai-labs"
                "/get-beam/main/get-beam.sh -sSfL | sh`."
            )
        self.app_creation()
        self.run_creation()

        process = subprocess.run(
            "beam deploy app.py", shell=True, capture_output=True, text=True
        )

        if process.returncode == 0:
            output = process.stdout
            logger.info(output)
            lines = output.split("\n")

            for line in lines:
                if line.startswith(" i  Send requests to: https://apps.beam.cloud/"):
                    self.app_id = line.split("/")[-1]
                    self.url = line.split(":")[1].strip()
                    return self.app_id

            raise ValueError(
                f"""Failed to retrieve the appID from the deployment output.
                Deployment output: {output}"""
            )
        else:
            raise ValueError(f"Deployment failed. Error: {process.stderr}")

    @property
    def authorization(self) -> str:
        if self.beam_client_id:
            credential_str = self.beam_client_id + ":" + self.beam_client_secret
        else:
            credential_str = self.beam_client_secret
        return base64.b64encode(credential_str.encode()).decode()

    def _call(
        self,
        prompt: str,
        stop: Optional[list] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call to Beam."""
        url = "https://apps.beam.cloud/" + self.app_id if self.app_id else self.url
        payload = {"prompt": prompt, "max_length": self.max_length}
        payload.update(kwargs)
        headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
            "Authorization": "Basic " + self.authorization,
            "Connection": "keep-alive",
            "Content-Type": "application/json",
        }

        for _ in range(DEFAULT_NUM_TRIES):
            request = requests.post(url, headers=headers, data=json.dumps(payload))
            if request.status_code == 200:
                return request.json()["text"]
            time.sleep(DEFAULT_SLEEP_TIME)
        logger.warning("Unable to successfully call model.")
        return ""
