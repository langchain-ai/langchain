import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from langchain.utilities.docker_containers import DockerContainer, DockerImage


def _get_dockerfile_content(base_image: str, script_path: str) -> str:
    return f"""FROM {base_image}
RUN pip install --no-cache-dir pydantic==1.10.12

RUN adduser -D runner
USER runner

WORKDIR /app

COPY {script_path} /app/python_runner.py
# Ensure python output is not buffered to remove logs delay
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
ENTRYPOINT ["python3", "/app/python_runner.py"]
"""


def _build_or_use_docker_image(
    base_image: str = "python:3.11-alpine3.18",
) -> DockerImage:
    """Builds docker image from python_docker_repl_runner.py script
    and docker template. Returns image object."""

    # we autogenerate deterministic name for the image
    name = f"langchain_pyrepl_{base_image}"
    import shutil
    from tempfile import TemporaryDirectory

    script_name = "python_docker_repl_runner.py"
    # workaround for https://github.com/docker/docker-py/issues/2105
    # which fails to use in-memory dockerfile with passing docker build
    # context. It requires to pass directory name.
    with TemporaryDirectory() as tmpdir:
        runner_script = Path(__file__).parent / script_name
        assert runner_script.exists()
        shutil.copy(runner_script, tmpdir)
        dockerfile_content = _get_dockerfile_content(base_image, script_name)
        dockerfile = Path(tmpdir) / "Dockerfile"
        with dockerfile.open("w") as f:
            f.write(dockerfile_content)
        return DockerImage.from_dockerfile(dockerfile.parent, name=name)


class PythonContainerREPL:
    """This class is a wrapper around the docker container that runs the python
    REPL server. It is used to execute python code in the container and return
    the results.

    It assumes specific docker image is used which runs langchain python runner
    server and it communicates by HTTP requests."""

    def __init__(
        self,
        port: int = 7123,
        image: Optional[DockerImage] = None,
        base_image: str = "python:3.11-alpine3.18",
        **kwargs: Dict[str, Any],
    ) -> None:
        """Starts docker container with python REPL server and wait till it
        gets operational.

        If image is not provided it will build based on
        the base_image and python_docker_repl_runner.py script.

        All other params: **kwargs are passed to DockerContainer constructor,
        however port mapping is hardcoded to map docker's 8080 to provided port.
        You can use it to limit memory/cpu etc. of the container.
        """
        # for now use the image we created.
        self.port = port
        if image is None and not base_image:
            raise ValueError("Either image or base_image must be provided.")
        self.image = (
            image if image is not None else _build_or_use_docker_image(base_image)
        )
        self.container = DockerContainer(self.image, ports={"8080/tcp": port}, **kwargs)
        # we need to start non-lexical scope lifetime for container
        # usually with statement should be used.
        # __del__ will close container.
        self.container.unsafe_start()
        self.session = requests.Session()
        # we need to ensure container is running and REPL server is
        # ready to accept requests, otherwise we might get connection
        # refused due to race conditions.
        self._wait_for_container_running()
        self._wait_for_repl_ready()

    def _wait_for_container_running(self, timeout: float = 3.0) -> None:
        status = self.container.docker_container.status
        while status not in ("created", "running"):
            time.sleep(0.1)
            timeout -= 0.1
            if timeout < 0:
                raise TimeoutError(f"Failed to start container - status={status}")

    def _wait_for_repl_ready(self, timeout: float = 3.0) -> None:
        while True:
            try:
                ex = self.session.get(f"http://localhost:{self.port}")
                if ex.text != "Hello! I am a python REPL server.":
                    raise Exception(
                        "Unrecognized banner, it is not a langchain python REPL server."
                    )
                break
            except Exception as ex:
                time.sleep(0.1)
                timeout -= 0.1
                if timeout < 0:
                    raise TimeoutError("Failed to boot service.")

    def __del__(self) -> None:
        self.container.unsafe_exit()

    def _exec(self, code: str, use_ast: bool = True) -> str:
        """Executes code and returns captured stdout. or error message."""
        import json

        try:
            msg = {"code": code, "use_ast": 1 if use_ast else 0}
            result = self.session.post(f"http://localhost:{self.port}", json=msg)
        except Exception as ex:
            return repr(ex.with_traceback(None))
        data = result.text
        if not data:
            return ""
        output = json.loads(data)
        return output.get("result", "")

    def eval(self, code: str) -> str:
        """Evaluate code and return result as string."""
        return self._exec(code, use_ast=True)

    def exec(self, code: str) -> str:
        """Execute code and return stdout."""
        return self._exec(code, use_ast=False)
