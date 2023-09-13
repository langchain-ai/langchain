from functools import lru_cache
from pathlib import Path
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import docker
import docker.errors

if TYPE_CHECKING:
    from docker.models.containers import Container


@lru_cache(maxsize=1)
def get_docker_client(**kwargs: Any) -> docker.DockerClient:
    """cached version to retrieve docker client. By default it will use environment
    variables to connect to docker daemon.
    """
    return docker.from_env(**kwargs)


def generate_langchain_container_tag() -> str:
    """Generates a random tag for a docker container."""
    import uuid
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return f"langchain_runner:{timestamp}-{uuid.uuid4().hex[:8]}"


class DockerImage:
    """Represents a locally available docker image as a tag.
    You can either use existing docker image or build a new one from Dockerfile.

    Examples:
    >>> image = DockerImage.from_tag("alpine")
    >>> image = DockerImage.from_tag("python", tag="3.9-slim")
    >>> image = DockerImage.from_dockerfile("example/Dockerfile")
    >>> image = DockerImage.from_dockerfile("path/to/dir_with_Dockerfile/", name="cow")
    """

    def __init__(self, name: str):
        """Note that it does not pull the image from the internet.
        It only represents a tag so it must exist on your system.
        It throws ValueError if docker image by that name does not exist locally.
        """
        splitted_name = name.split(":")
        if len(splitted_name) == 1:
            # by default, image has latest tag.
            self.name = name + ":latest"
        else:
            self.name = name

        if not self.exists(name):
            raise ValueError(
                f"Invalid value: name={name} does not exist on your system."
                "Use DockerImage.from_tag() to pull it."
            )

    def __repr__(self) -> str:
        return f"DockerImage(name={self.name})"

    @classmethod
    def exists(cls, name: str) -> bool:
        """Checks if the docker image exists"""
        docker_client = get_docker_client()
        return len(docker_client.images.list(name=name)) > 0

    @classmethod
    def remove(cls, name: str) -> None:
        """WARNING: Removes image from the system, be cautious with this function.
        It is irreversible operation!.
        """
        if cls.exists(name):
            docker_client = get_docker_client()
            docker_client.images.remove(name)

    @classmethod
    def from_tag(
        cls,
        repository: str,
        tag: str = "latest",
        auth_config: Optional[Dict[str, str]] = None,
    ) -> "DockerImage":
        """Use image with a given repository and tag. It is going to pull it if it is
        not present on the system.
        Example: repository = "alpine" (will get "latest" tag)
        Example: repository = "python" tag = "3.9-slim"
        """
        docker_client = get_docker_client()
        name = f"{repository}:{tag}"
        if len(docker_client.images.list(name=name)) > 0:
            return cls(name=name)
        docker_client.images.pull(
            repository=repository, tag=tag, auth_config=auth_config
        )
        return cls(name=name)

    @classmethod
    def from_dockerfile(
        cls,
        dockerfile_path: Union[Path, str],
        name: Union[str, Callable[[], str]] = generate_langchain_container_tag,
        **kwargs: Any,
    ) -> "DockerImage":
        """Build a new image from Dockerfile given its file path."""

        img_name = (
            name
            if isinstance(name, str) and name
            else generate_langchain_container_tag()
        )
        dockerfile = Path(dockerfile_path)

        docker_client = get_docker_client()

        if dockerfile.is_dir():
            if not (dockerfile / "Dockerfile").exists():
                raise ValueError(
                    f"Directory {dockerfile} does not contain a Dockerfile."
                )
            docker_client.images.build(
                path=str(dockerfile), tag=img_name, rm=True, **kwargs
            )
        elif dockerfile.name == "Dockerfile" and dockerfile.is_file():
            docker_client.images.build(
                fileobj=dockerfile.open("rb"), tag=img_name, rm=True, **kwargs
            )
        else:
            raise ValueError(f"Invalid parameter: dockerfile: {dockerfile}")

        return cls(name=img_name)


class DockerContainer:
    """An isolated environment for running commands, based on docker container.

    Examples:
    If you need to run container for a single job:
    >>> container = DockerContainer(DockerImage.from_tag("alpine"))
    >>> status_code, logs = container.spawn_run("echo hello world")

    To run a container in background and execute commands:
    >>> with DockerContainer(DockerImage.from_tag("alpine")) as container:
    >>>     status_code, logs = container.run("echo hello world")
    """

    def __init__(self, image: DockerImage, **kwargs: Any):
        """Wraps docker image to control container interaction.
        NOTE: **kwargs are passed to docker client containers.run() method so you can
        use them as you wish.
        """
        self.image = image
        self._client = get_docker_client()
        self._container = None
        self._run_kwargs = kwargs

    def __enter__(self) -> "DockerContainer":
        """Enters container context. It means that container is started and you can
        execute commands inside it.
        """
        self.unsafe_start()
        return self

    def unsafe_start(self) -> None:
        """Starts container without entering it.
        Please prefer to use with DockerContainer statement.
        """
        assert self._container is None, "You cannot re-entry container"
        # tty=True is required to keep container alive
        self._container = self._client.containers.run(
            self.image.name,
            detach=True,
            tty=True,
            **self._run_kwargs,
        )

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        """Cleanup container on exit."""
        assert self._container is not None, "You cannot exit unstarted container."
        if exc_type is not None:
            # re-throw exception. try to stop container and remove it
            try:
                self._cleanup()
            except Exception as e:
                print("Failed to stop and remove container to cleanup exception.", e)
            return False
        else:
            self.unsafe_exit()
            return True

    def unsafe_exit(self):
        """Cleanup container on exit. Please prefer to use `with` statement."""
        if self._container is None:
            return
        self._cleanup()
        self._container = None

    def spawn_run(
        self, command: Union[str, List[str]], **kwargs: Any
    ) -> Tuple[int, bytes]:
        """Run a script in the isolated environment which is docker container with the
        same lifetime as this function call.

        You can also pass all arguments that docker client containers.run() accepts.
        It blocks till command is finished.
        """
        # we can update here kwargs with self._run_kwargs so user can override them
        custom_kwargs = (
            self._run_kwargs.copy().update(kwargs) if kwargs else self._run_kwargs
        )
        # There is a known issue with auto_remove=True and docker-py:
        # https://github.com/docker/docker-py/issues/1813
        # so as workaround we detach, wait & and remove container manually
        container = self._client.containers.run(
            self.image.name, command=command, detach=True, **custom_kwargs
        )
        status_code = container.wait().get("StatusCode", 1)
        logs = container.logs()
        container.remove()
        return status_code, logs

    @property
    def docker_container(self) -> Container:
        """Returns docker container object."""
        assert (
            self._container is not None
        ), "You cannot access container that was not entered"
        return self._container

    @property
    def name(self) -> str:
        """Name of the container if it exists, empty string otherwise."""
        if self._container:
            return self._container.name
        return ""

    def run(
        self, command: Union[str, List[str]], **kwargs: Any
    ) -> Tuple[int, Union[bytes, Tuple[bytes, bytes], Generator[bytes, None, None]]]:
        """Run a script in the isolated environment which is docker container.
        You can send any args which docker-py exec_run accepts:
        https://docker-py.readthedocs.io/en/stable/containers.html#docker.models.containers.Container.exec_run
        Return is a tuple of exit code and output which is controlled by arguments:
        stream, socket and demux.
        """
        assert (
            self._container is not None
        ), "You cannot execute command in container that was not entered"

        exit_code, output = self._container.exec_run(cmd=command, **kwargs)
        return exit_code, output

    def _cleanup(self) -> None:
        """Stops and removes container."""
        if self._container is None:
            return
        self._container.stop()
        self._container.remove()
