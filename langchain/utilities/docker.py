"""Wrapper for untrusted code exectuion on docker."""
# TODO: Validation:
#       - verify gVisor runtime (runsc) if available
#       - pass arbitrary image names

import docker
from docker.client import DockerClient  # type: ignore
from docker.errors import APIError, ContainerError

from typing import Any, Dict
from typing import Optional
from pydantic import BaseModel, PrivateAttr, Extra, root_validator, validator


class DockerWrapper(BaseModel, extra=Extra.forbid):
    """Executes arbitrary payloads and returns the output."""

    _docker_client: DockerClient = PrivateAttr()
    image: Optional[str] = "alpine"

    # use env by default when create docker client
    from_env: Optional[bool] = True

    def __init__(self, **kwargs):
        """Initialize docker client."""
        super().__init__(**kwargs)

        if self.from_env:
            self._docker_client = docker.from_env()

    @property
    def client(self) -> DockerClient:
        """Docker client."""
        return self._docker_client

    @property
    def info(self) -> Any:
        """Prints docker `info`."""
        return self._docker_client.info()

    @root_validator()
    def validate_all(cls, values: Dict) -> Dict:
        """Validate environment."""
        # print("root validator")
        return values

    def run(self, query: str, **kwargs: Any) -> str:
        """Run arbitrary shell command inside a container.

        Args:
            **kwargs: Pass extra parameters to DockerClient.container.run.

        """
        try:
            image = getattr(kwargs, "image", self.image)
            return self._docker_client.containers.run(image,
                                                      query,
                                                      remove=True)
        except ContainerError as e:
            return f"STDERR: {e}"
        # TODO: handle docker APIError ?
