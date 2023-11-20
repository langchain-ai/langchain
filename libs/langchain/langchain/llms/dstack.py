import copy
import logging
from typing import Any, Dict, List, Optional, cast

import requests

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.load.serializable import Serializable
from langchain.pydantic_v1 import Field, SecretStr, root_validator
from langchain.utils import convert_to_secret_str, get_from_dict_or_env

logger = logging.getLogger(__name__)


class _BaseDstack(Serializable):
    run_name: str = ""
    """dstack run_name to connect to"""
    api_base_url: str = ""
    """dstack server url"""
    api_token: SecretStr = SecretStr("")
    """dstack server token"""
    project: str = ""
    """dstack project"""
    service_url: str = ""
    """dstack service url"""

    parameters: Dict[str, Any] = Field(default_factory=dict)
    """TGI /generate parameters"""

    tokenizer: Any = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Load credentials from environment variables.
        Prepare tokenizer if `transformers` is installed.
        """

        if not values["service_url"]:
            values["api_base_url"] = get_from_dict_or_env(
                values, "api_base_url", "DSTACK_API_BASE_URL"
            )
            values["api_token"] = convert_to_secret_str(
                get_from_dict_or_env(values, "api_token", "DSTACK_API_TOKEN")
            )
            values["project"] = get_from_dict_or_env(
                values, "project", "DSTACK_PROJECT"
            )

            try:
                run = _dstack_get_run(
                    values["api_base_url"],
                    values["api_token"],
                    values["project"],
                    values["run_name"],
                )
            except requests.RequestException:
                raise ValueError(
                    "Failed to fetch a run from the dstack server,"
                    " please check your credentials"
                )

            if len(run["jobs"]) > 1:
                raise ValueError("Too many jobs")
            job = run["jobs"][0]

            configuration_type = run["run_spec"]["configuration"]["type"]
            if configuration_type == "service":
                gateway = job["job_spec"]["gateway"]
                schema = "https" if gateway["secure"] else "http"
                values[
                    "service_url"
                ] = f"{schema}://{gateway['hostname']}:{gateway['public_port']}"
            else:
                raise ValueError(f"The run type is not supported: {configuration_type}")

        try:
            from transformers import AutoTokenizer

            info = _dstack_tgi_info(values["service_url"])
            tokenizer = AutoTokenizer.from_pretrained(info["model_id"])
            _ = tokenizer.apply_chat_template  # test if presented
            values["tokenizer"] = tokenizer
        except ImportError:
            logger.warning(
                "Transformers is not installed, you won't be able to use chat template"
            )
        except AttributeError:
            logger.warning(
                "Transformers is installed, but it's too old,"
                " you won't be able to use chat template"
            )
        except requests.RequestException:
            raise ValueError(
                "Failed to fetch info from the service, check if the service is running"
            )

        return values

    @property
    def _llm_type(self) -> str:
        return "dstack_service"

    def _tgi_parameters(
        self, stop: Optional[List[str]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Inject stop_sequences into parameters.

        Returns:
            Parameters for /generate endpoint
        """
        parameters = copy.deepcopy(self.parameters)
        parameters.update(kwargs)
        parameters["stop_sequences"] = parameters.get("stop_sequences", []) + (
            stop or []
        )
        return parameters

    def _tgi_generate(self, inputs: str, parameters: Dict[str, Any]) -> Dict:
        """Call the service.

        Returns:
            Model output
        """
        resp = requests.post(
            f"{self.service_url.rstrip('/')}/generate",
            json={
                "inputs": inputs,
                "parameters": parameters,
            },
        )
        resp.raise_for_status()
        return resp.json()


class Dstack(_BaseDstack, LLM):
    """dstack services API

    To use chat templating, you should have `transformers` library.

    Required parameters:
        - `run_name`
        - `api_base_url` or in the environment variable `DSTACK_API_BASE_URL`
        - `api_token` or in the environment variable `DSTACK_API_TOKEN`
        - `project` or in the environment variable `DSTACK_PROJECT`

    Example:
        .. code-block:: python
            from langchain.llms import Dstack
            llm = Dstack(run_name="mistral-7b-gptq")
    """

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the service and return the generated text.

        Args:
            prompt: The prompt to pass to the model.
            stop: List of stop sequences.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python
                llm = Dstack(run_name="mistral-7b-gptq")
                response = llm("Write a poem.")
        """
        if self.tokenizer is not None:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False
            )
            logger.debug("Prompt after chat templating: %s", prompt)

        resp = self._tgi_generate(prompt, self._tgi_parameters(stop, **kwargs))
        return resp["generated_text"]


def _dstack_get_run(server_url: str, token: str, project: str, run_name: str) -> Dict:
    """Get run from the dstack server.

    Returns:
        Run info
    """
    resp = requests.post(
        f"{server_url.rstrip('/')}/api/project/{project}/runs/get",
        headers={
            "Authorization": f"Bearer {cast(SecretStr, token).get_secret_value()}"
        },
        json={"run_name": run_name},
    )
    resp.raise_for_status()
    return resp.json()


def _dstack_tgi_info(service_url: str) -> Dict:
    """Get deployed model info from the service.

    Returns:
        Model info
    """
    resp = requests.get(f"{service_url.rstrip('/')}/info")
    resp.raise_for_status()
    return resp.json()
