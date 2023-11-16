import copy
from typing import Optional, List, Any, Dict, cast

import requests

from langchain.pydantic_v1 import Field, SecretStr, root_validator
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.utils import convert_to_secret_str, get_from_dict_or_env


class Dstack(LLM):
    """
    dstack gateway API

    To use, provider credentials in environment variables: `DSTACK_SERVER_URL`, `DSTACK_TOKEN`, `DSTACK_PROJECT`

    TODO example
    """

    run_name: str
    """dstack run_name to connect to"""
    dstack_server_url: Optional[str] = None
    """dstack server url"""
    dstack_token: Optional[SecretStr] = None
    """dstack token"""
    dstack_project: Optional[str] = None
    """dstack project"""
    dstack_service_url: Optional[str] = None
    """dstack service url"""

    parameters: Dict[str, Any] = Field(default_factory=dict)
    """TGI /generate parameters"""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Load credentials from environment variables."""

        values["dstack_server_url"] = get_from_dict_or_env(values, "dstack_server_url", "DSTACK_SERVER_URL")
        values["dstack_token"] = convert_to_secret_str(get_from_dict_or_env(values, "dstack_token", "DSTACK_TOKEN"))
        values["dstack_project"] = get_from_dict_or_env(values, "dstack_project", "DSTACK_PROJECT")

        if values["dstack_service_url"] is None:
            try:
                run = _dstack_get_run(values["dstack_server_url"], values["dstack_token"], values["dstack_project"], values["run_name"])
            except requests.RequestException as e:
                raise ValueError("Failed to fetch a run from the dstack server, please check your credentials")

            if len(run["jobs"]) > 1:
                raise ValueError("Too many jobs")
            job = run["jobs"][0]

            # TODO test if status == running

            configuration_type = run["run_spec"]["configuration"]["type"]
            if configuration_type == "service":
                gateway = job["job_spec"]["gateway"]
                schema = "https" if gateway["secure"] else "http"
                values["dstack_service_url"] = f"{schema}://{gateway['hostname']}:{gateway['public_port']}"
            else:
                raise ValueError(f"The run type is not supported: {configuration_type}")

        return values

    def _call_parameters(self, stop: Optional[List[str]], **kwargs) -> Dict[str, Any]:
        parameters = copy.deepcopy(self.parameters)
        parameters.update(kwargs)
        parameters["stop_sequences"] = parameters.get("stop_sequences", []) + (stop or [])
        return parameters

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        resp = _dstack_tgi_generate(self.dstack_service_url, self._call_parameters(stop, **kwargs), prompt)
        return resp["generated_text"]

    # TODO async support

    @property
    def _llm_type(self) -> str:
        return "dstack_gateway"


def _dstack_get_run(server_url: str, token: str, project: str, run_name: str) -> Dict:
    resp = requests.post(
        f"{server_url.rstrip('/')}/api/project/{project}/runs/get",
        headers={"Authorization": f"Bearer {cast(SecretStr, token).get_secret_value()}"},
        json={"run_name": run_name}
    )
    resp.raise_for_status()
    return resp.json()


def _dstack_tgi_generate(service_url: str, parameters: Dict, prompt: str) -> Dict:
    resp = requests.post(f"{service_url.rstrip('/')}/generate", json={
        "inputs": prompt,
        "parameters": parameters,
    })
    resp.raise_for_status()
    return resp.json()
