"""Wrapper around Beam API."""
import base64
import json
import logging
import subprocess
import textwrap
import time
from typing import Any, Dict, List, Mapping, Optional, Union, cast

import requests
from pydantic import Extra, Field, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class Beam(LLM):
    """Wrapper around Beam API for gpt2 large language model.

    To use, you should have the ``beam-sdk`` python package installed,
    and the environment variable ``beam_client_id`` set with your client id
    and ``beam_client_secret`` set with your client secret. Information on how
    to set these is availible here: https://docs.beam.cloud/account/api-keys.

    The wrapper can then be called as follows, where the name, cpu, memory, gpu,
    python version, and python packages can be updated accordingly. Once deployed,
    the instance can be called.
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

        depolyResult = llm._deploy(prompt=input)

        callResult = llm._call(prompt=input)
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
                    f"""{field_name} was transfered to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

    @root_validator()
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
            **{"model_name": self.model_name},
            **{"name": self.name},
            **{"cpu": self.cpu},
            **{"memory": self.memory},
            **{"gpu": self.gpu},
            **{"python_version": self.python_version},
            **{"python_packages": self.python_packages},
            **{"max_length": self.max_length},
            **{"model_kwargs": self.model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return self.model_name

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

            print(output)
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

            if (beam.__path__) == "":
                raise ImportError
        except ImportError:
            raise ValueError(
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
            print(process.stdout)
            output = process.stdout
            lines = output.split("\n")
            app_id = None

            for line in lines:
                if line.startswith(" i  Send requests to: https://apps.beam.cloud/"):
                    app_id = line.split("/")[-1]
                    self.url = line.split(":")[1].strip()
                    return app_id

            raise ValueError(
                f"""Failed to retrieve the appID from the deployment output.
                Error: {process.stdout}"""
            )
        else:
            raise ValueError(f"Deployment failed. Error: {process.stderr}")

    def _call(
        self,
        prompt: str,
        stop: Optional[list] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call to Beam."""
        try:
            import beam  # type: ignore

            if (beam.__path__) == "":
                raise ImportError
        except ImportError:
            raise ValueError(
                "Could not import beam python package. "
                "Please install it with `curl "
                "https://raw.githubusercontent.com/slai-labs"
                "/get-beam/main/get-beam.sh -sSfL | sh`."
            )

        api_key = self.beam_client_id
        api_secret = self.beam_client_secret
        max_length = self.max_length
        response = ""
        accept = "*/*"
        encoding = "gzip, deflate"
        credential_string = (
            (api_key + ":" + api_secret) if api_key is not None else api_secret
        )
        authorization = base64.b64encode(credential_string.encode()).decode()
        connection = "keep-alive"
        content_type = "application/json"
        app_id = kwargs.get("app_id")
        if app_id is not None:
            url = "https://apps.beam.cloud/" + app_id
        else:
            url = self.url
        payload = {"prompt": prompt, "max_length": max_length}
        headers = {
            "Accept": accept,
            "Accept-Encoding": encoding,
            "Authorization": "Basic " + authorization,
            "Connection": connection,
            "Content-Type": content_type,
        }

        completed = False
        tries = 0
        while not completed and tries < 100:
            request = requests.request(
                "POST",
                cast(Union[str, bytes], url),
                headers=headers,
                data=json.dumps(payload),
            )
            if request.status_code == 200:
                response = request.json()["text"]
                completed = True
            else:
                time.sleep(10)
                tries += 1

        return response
