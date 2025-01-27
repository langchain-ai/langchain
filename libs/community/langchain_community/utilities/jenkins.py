"""Wrapper for the Jenkins API"""

import time
from typing import Any, Dict, Optional

from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, model_validator


class JenkinsAPIWrapper(BaseModel):
    """Wrapper for Jenkins API

    To use, set the environment variables ``JENKINS_SERVER``,
    ``USERNAME`` and ``PASSWORD``. OR those input can supplay by API parameter."""

    jenkins_client: Any

    jenkins_server: Optional[str]
    username: Optional[str]
    password: Optional[str]

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        jenkins_server = get_from_dict_or_env(
            values, "jenkins_server", "JENKINS_SERVER"
        )
        values["jenkins_server"] = jenkins_server

        username = get_from_dict_or_env(values, "username", "USERNAME")
        values["username"] = username

        password = get_from_dict_or_env(values, "password", "PASSWORD")
        values["password"] = password

        try:
            from jenkins import Jenkins
        except ImportError:
            raise ImportError(
                """jenkins package not found, 
                please install it with pip install python-jenkins"""
            )

        jenkins_client = Jenkins(jenkins_server, username=username, password=password)
        values["jenkins_client"] = jenkins_client

        return values

    def delete_job(self, job: str) -> Any:
        try:
            return self.jenkins_client.delete_job(job)
        except Exception:
            pass

    def create_job(self, job: str, config_xml: str) -> Any:
        return self.jenkins_client.create_job(job, config_xml)

    def run_job(self, job: str, parameters: dict) -> int:
        next_build_number = self.jenkins_client.get_job_info(job)["nextBuildNumber"]
        self.jenkins_client.build_job(job, parameters=parameters)
        return next_build_number

    def status_job(self, job: str, number: int) -> Any:
        from jenkins import JenkinsException, NotFoundException

        try:
            return self.jenkins_client.get_build_info(job, number)
        except (NotFoundException, JenkinsException):
            time.sleep(5)
