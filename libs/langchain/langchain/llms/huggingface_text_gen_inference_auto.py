"""
Spin up a HuggingFace TextGen Inference server automatically from within Python.

Note: This code isn't really intended for production use. Intent is to facilitate
easy rapid prototyping of LLM powered applications by automating some of the docker
inference server startup / container state management steps. Controlling docker
from within python tends to be flaky; the goal of this feature though is to make
it super easy to try out different LLMs  during development. Once happy with the
performance, swap back over to using the same model with the HF textgen server
directly / Langchain HuggingFaceTextGenInference wrapper directly.

```python
model_name = 'bigscience/bloom-560m'
llm = HuggingFaceTextGenInferenceAuto.from_docker(
    model_name,
    host='0.0.0.0',
    port=8080,
    shm_size="1g"
)
answer = llm("How old is the universe?")
print(answer)
```

"""
import hashlib
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import time
import warnings
from functools import wraps
from pathlib import Path
from typing import Callable, List, Optional, Tuple, TypeVar, Union

import requests

from langchain.llms import HuggingFaceTextGenInference
from langchain.pydantic_v1 import BaseModel, Field, root_validator

try:
    import docker
    from docker.models.containers import Container
except ImportError:
    # We want to make this module entirely optional, so allow module
    # to be imported regardless of whether user has python docker
    # library installed. We'll raise an error at runtime if they try
    # to actually use any of the methods in the
    # AutoHuggingFaceTextGenInference class.
    docker = None
    Container = None


# This will be prefixed to container names, e.g. <service-name>--...
SERVICE_NAME = "text-generation-inference"


def is_debian_based() -> bool:
    """
    Check if the system is Debian-based.
    Returns True if the system is Debian-based, otherwise False.
    """
    return os.path.exists("/etc/debian_version")


def check_os() -> Tuple[bool, str]:
    """
    Check if the operating system is compatible.
    Returns a tuple containing a boolean and a message indicating the compatibility.
    """
    os_type = platform.system().lower()
    if os_type == "darwin":
        if platform.machine() == "arm64":
            return (
                False,
                f"{__name__} module is not compatible with macOS with Apple Silicon.",
            )
        else:
            return (
                True,
                f"{__name__} module hasn't been explicitly tested on macOS "
                f"without Apple Silicon. YMMV.",
            )
    elif os_type == "windows":
        return True, f"{__name__} hasn't been tested on Windows. YMMV."
    elif os_type == "linux":
        if not is_debian_based():
            return (
                True,
                f"{__name__} module hasn't been explicitly tested with non-Debian "
                f"based Linux distros. YMMV.",
            )
        else:
            return True, ""
    else:
        return False, f"{__name__} module is not compatible with {os_type}"


def check_gpu() -> bool:
    """
    Check if an NVIDIA GPU is available.
    Returns True if an NVIDIA GPU is found, otherwise False.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return "NVIDIA" in result.stdout
    except FileNotFoundError:
        return False


def check_nvidia_docker() -> bool:
    """
    Check if the NVIDIA Docker runtime is installed.
    Returns True if the NVIDIA Docker runtime is found, otherwise False.
    """
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.Runtimes.nvidia}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return "nvidia" in result.stdout
    except FileNotFoundError:
        return False


def check_prerequisites() -> Tuple[bool, str]:
    """
    Check all prerequisites for using AutoHuggingFaceTextGenInference.
    Returns a tuple containing a boolean and a message indicating whether
    the prerequisites are met.
    """
    if docker is None:
        docker_msg = (
            "Python `docker` library not found in current environment. "
            "`pip install docker` if you want to use AutoHuggingFaceTextGenInference."
        )
        return False, docker_msg

    os_check, os_message = check_os()
    if not os_check:
        return False, os_message
    if not check_gpu():
        return False, "No NVIDIA GPU found."
    if not check_nvidia_docker():
        return False, "NVIDIA Docker runtime is not installed."

    return True, os_message


def _validate_environment() -> None:
    """Perform environment check for using this module."""
    is_compatible, compatibility_msg = check_prerequisites()
    if not is_compatible:
        raise EnvironmentError(compatibility_msg)
    elif is_compatible and compatibility_msg != "":
        # Let the user give things a try at their own risk
        warnings.warn(compatibility_msg)


# Define a type variable to represent the return type of decorated function
RT = TypeVar("RT")


def validate_environment(func: Callable[..., RT]) -> Callable[..., RT]:
    """Decorator for performing environment check."""

    @wraps(func)
    def wrapper(cls, *args, **kwargs) -> RT:
        _validate_environment()
        return func(cls, *args, **kwargs)

    return wrapper


def bytes2human(n: float, format="%(value).1f %(symbol)s", symbols="customary") -> str:
    """
    Convert n bytes into a human readable string based on format.
    symbols can be either "customary", "customary_ext", "iec" or "iec_ext",
    see: http://goo.gl/kTQMs
    """
    SYMBOLS = {
        "customary": ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"),
        "customary_ext": (
            "byte",
            "kilo",
            "mega",
            "giga",
            "tera",
            "peta",
            "exa",
            "zetta",
            "iotta",
        ),
        "iec": ("Bi", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi", "Yi"),
        "iec_ext": (
            "byte",
            "kibi",
            "mebi",
            "gibi",
            "tebi",
            "pebi",
            "exbi",
            "zebi",
            "yobi",
        ),
    }
    n = int(n)
    if n < 0:
        raise ValueError("n < 0")
    symbols = SYMBOLS[symbols]
    prefix = {}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i + 1) * 10
    for symbol in reversed(symbols[1:]):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return format % locals()
    return format % dict(symbol=symbols[0], value=n)


class DockerPath(pathlib.Path):
    """
    A subclass of pathlib.Path for interacting with a Docker container's filesystem.

    This class overrides the exists(), is_dir(), and is_file() __truediv__ methods
    to operate on the specified Docker container's filesystem.

    """

    _flavour = type(pathlib.Path())._flavour

    def __new__(cls, container_id: str, *pathsegments):
        self = super().__new__(cls, *pathsegments)
        self.container_id = container_id  # moved this line here
        self.client = docker.from_env()
        self.container = self.client.containers.get(container_id)
        return self

    def __truediv__(self, key: Union[str, os.PathLike[str]]):
        return self.__class__(self.container_id, self, key)

    def _run_command(self, command: str):
        result = self.container.exec_run(f'/bin/sh -c "{command}"')
        return result.output.decode("utf-8").strip()

    def exists(self):
        command = f"test -e {self} && echo exists || echo does not exist"
        output = self._run_command(command)
        return output == "exists"

    def is_dir(self):
        command = f"test -d {self} && echo is_dir || echo not_dir"
        output = self._run_command(command)
        return output == "is_dir"

    def is_file(self):
        command = f"test -f {self} && echo is_file || echo not_file"
        output = self._run_command(command)
        return output == "is_file"

    def stat(self, *, follow_symlinks: bool = ...) -> os.stat_result:
        """
        Retrieve file size information for the specified path on the Docker container.

        Note: Even though we return an instance of os.stat_result, everything
        but except the `st_size` attribute is a dummy!

        Returns:
            stat_result   os.stat_result
        """
        command = f'stat -c "%s" {self}'

        output = self._run_command(command)
        try:
            size = int(output)
            s = [0, 0, 0, 0, 0, 0, size, 0, 0, 0]
            stat_result = os.stat_result(s)
            return stat_result
        except ValueError:
            stat_result = os.stat_result([-1] * 10)
            return stat_result


class CountdownProgressBar:
    """
    Simple countdown progress bar with functionality similar to tqdm.

    Useful for things like timeout countdowns, etc.
    """

    def __init__(self, total, bar_length=50, desc="Progress"):
        self.total = total
        self.bar_length = bar_length
        self.desc = desc
        self.current = total

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Optionally, can handle exceptions here;
        pass

    def update(self, decrement=1, new_desc=None):
        self.current -= decrement
        if new_desc is not None:
            self.desc = new_desc
        filled_length = int(self.bar_length * self.current / self.total)
        bar = "#" * filled_length + "-" * (self.bar_length - filled_length)
        sys.stdout.write(f"\r{self.desc}: {bar} {self.current}/{self.total} ")
        sys.stdout.flush()


def get_directory_size(path: Union[str, Path]) -> int:
    """Get total size of directory (recursively) in bytes."""
    path = str(path)
    total_size = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    return total_size


def is_valid_hf_model_name_format(model_name: str) -> bool:
    """Verify model name follows HuggingFace format model-owner/model-name"""
    # The regex pattern for a valid model name format
    pattern = re.compile(r"^[a-zA-Z0-9-_]+\/[a-zA-Z0-9-_]+$")
    return bool(pattern.match(model_name))


def hf_model_name_to_model_dir(name: str) -> str:
    """Get the model directory a HuggingFace model will download in.

    This is based on the default behavior of the HuggingFace Hub / Transformers API.
    A model like will get downloaded like:
        <owner>/<model-id> --> models--<owner>--<model-id>

    """
    n = name.split("/")
    if len(n) != 2:
        raise ValueError(
            f"Model name '{name}' is not a valid HF model name. "
            f"Expected format like: <owner>/<model-id>"
        )
    owner, model_id = n
    return f"models--{owner}--{model_id}"


def get_image_tag(image: str):
    """Return tag from docker image_name:tag"""
    if ":" not in image:
        return ""
    else:
        try:
            tag = image.split(":")[1]
            return tag
        except IndexError:
            return ""


def convert_shm_size(shm_size_str: str) -> int:
    """
    Convert shared memory size from string (e.g., "1g") to integer (e.g., 1073741824).

    Args:
        shm_size_str (str): Shared memory size as a string.

    Returns:
        int: Shared memory size as an integer.
    """
    if shm_size_str.isdigit():
        return int(shm_size_str)
    if shm_size_str.endswith("g") or shm_size_str.endswith("G"):
        return int(shm_size_str[:-1]) * 1024 * 1024 * 1024
    if shm_size_str.endswith("m") or shm_size_str.endswith("M"):
        return int(shm_size_str[:-1]) * 1024 * 1024
    if shm_size_str.endswith("k") or shm_size_str.endswith("K"):
        return int(shm_size_str[:-1]) * 1024
    raise ValueError(f"Invalid shm_size format: {shm_size_str}")


def get_container(client, name: str) -> Union[Container, None]:
    """
    Retrieve a container with a given name.

    :param name: name of the container.
    :return: The docker Container object or None if no matching container is found.
    """
    # List all containers, including those that are not running
    containers = client.containers.list(all=True)

    for container in containers:
        if container.name == name:
            return container

    return None


def get_model_name_from_container(container: Container) -> str:
    # Get model name from container args. Since this always has to be provided
    # to start up the textgen-inference container, safe to assume it's always available
    cmd_args: list = container.attrs["Args"]
    i = cmd_args.index("--model-id")
    model_name = cmd_args[i + 1]
    return model_name


def stop_container(client, name: str, timeout=30) -> None:
    """
    Stop a container using its ID, short ID, or name.

    :param uid: The ID, short ID, or name of the container.
    """
    container = get_container(client, name)
    if container is not None:
        print(f"Stopping container for {name}")
        container.stop()
        container.wait(timeout=timeout, condition="removed")
        container = get_container(client, name)
        if not container:
            return
        else:
            raise ValueError(
                "Docker container did not shut down properly "
                "(or didn't shut down before timeout)"
            )
    else:
        print(f"No running container found for {name}")

    if not container:
        return


def is_host_port_available(client, port: int) -> bool:
    for container in client.containers.list():
        ports = container.ports
        for _, bindings in ports.items():
            if bindings:
                for binding in bindings:
                    if binding["HostPort"] == str(port):
                        return False
    return True


def find_available_host_port(
    client,
    desired_port: int,
    auto_increment=True,
    max_retries=10,
    port_range=(1024, 65535),
) -> Union[int, None]:
    if port_range[0] < 1024 or port_range[1] > 65535:
        raise ValueError(
            f"Port range must be in range 1024-65535, inclusive. Got: {port_range}"
        )

    if not auto_increment:
        return desired_port if is_host_port_available(client, desired_port) else None

    port = desired_port
    retries = 0

    while not is_host_port_available(client, port):
        if retries >= max_retries or port >= port_range[1]:
            raise ValueError(
                f"No available port found within {max_retries} "
                f"retries or port range {port_range}."
            )
        port += 1
        retries += 1

    return port


def start_model_container(
    client,
    pretrained_model_name: str,
    container_name: str,
    volume_path: Union[str, Path],
    port: int,
    shm_size: int,
    image_tag: str = "1.1.0",
) -> Container:
    volume_path = volume_path if isinstance(volume_path, str) else str(volume_path)

    print(f"Starting container: '{container_name}'")
    host_port = port
    container_port = 80  # Hardcoding this
    container = client.containers.run(
        f"ghcr.io/huggingface/text-generation-inference:{image_tag}",
        f"--model-id {pretrained_model_name}",
        name=container_name,
        volumes={volume_path: {"bind": "/data", "mode": "rw"}},
        ports={f"{container_port}/tcp": host_port},
        shm_size=shm_size,
        detach=True,
        runtime="nvidia",
        remove=True,  # Hardcoding this makes state management easier for us
    )

    return container


class _SharedParams(BaseModel):
    """Params in common for Docker and HuggingFaceTextGenIngerence."""

    port: int  # This should be the port on the HOST machine

    def to_dict(self, **kwargs) -> dict:
        exclude = kwargs.get("exclude", set())
        return super().dict(exclude=exclude)


class DockerParams(_SharedParams):
    pretrained_model_name: str  # Pydantic doesn't like things prefixed with 'model'
    volume_path: Union[str, Path]
    image_tag: str = "1.1.0"
    shm_size: Optional[Union[int, str]] = 1073741824  # 1gb in bytes

    @property
    def container_name(self) -> str:
        """Deterministic name for the container."""
        param_hash = self.compute_md5()
        model_owner, model_id = self.pretrained_model_name.split("/")
        name = f"{SERVICE_NAME}--{model_owner}--{model_id}--{param_hash}"
        return name

    @root_validator
    def _validate(cls, values):
        """
        Python docker library handles shm_size in bytes, so if string like
        e.g. '1g', '512m' is passed in, convert it to bytes.
        """

        if not os.path.isabs(values["volume_path"]):
            raise ValueError(
                f"Volume path must be absolute. Got: {values['volume_path']}"
            )
        values["volume_path"] = str(values["volume_path"])

        if not is_valid_hf_model_name_format(values["pretrained_model_name"]):
            raise ValueError(
                f"'{values['pretrained_model_name']}' is not valid HuggingFace name"
                "Model name should be formatted like 'model-owner/model-name'"
            )

        # Python docker library handles shm_size in bytes, so if string like
        # e.g. '1g', '512m' is passed in, convert it to bytes.
        for key, val in values.items():
            if key == "shm_size":
                if isinstance(val, str):
                    values["shm_size"] = convert_shm_size(val)
        return values

    @classmethod
    def from_container(cls, container: Container) -> "DockerParams":
        """Given an existing container, get the params used to construct it."""
        c = container.attrs

        # Get model name
        model_name = get_model_name_from_container(container)

        # Get volume details
        # This means identity will be formulated based on first volume only;
        # think that's okay for our use case here.
        # Also means we implicitly requires that a bind volume always be present.
        binds = c["HostConfig"]["Binds"][0]
        volume_path = binds.split(":")[0]

        # Get ports
        host_port = int(c["HostConfig"]["PortBindings"]["80/tcp"][0]["HostPort"])

        # Get shared memory size and misc items
        shm_size = c["HostConfig"]["ShmSize"]
        image_tag = get_image_tag(c["Config"]["Image"])

        return DockerParams(
            port=host_port,
            pretrained_model_name=model_name,
            volume_path=volume_path,
            image_tag=image_tag,
            shm_size=shm_size,
        )

    def compute_md5(self) -> str:
        """Compute MD5 hash based on parameter values."""
        params_dict = self.to_dict()
        params_str = json.dumps(
            params_dict, sort_keys=True
        )  # Serialize to JSON with sorted keys for consistency
        md5_hash = hashlib.md5(params_str.encode()).hexdigest()
        return md5_hash

    def equals_with_exclusions(
        self, other: "DockerParams", exclude: Union[str, List[str]] = ""
    ) -> bool:
        """Compare parameters, excluding those specified by key or list of keys."""
        exclude = [] if exclude == "" else exclude
        exclude = [exclude] if isinstance(exclude, str) else exclude
        if not exclude and self == other:
            # Nothing to exclude so, this case is equivalent to __eq__
            return True

        # Convert both instances to dictionaries
        self_dict = self.to_dict()
        other_dict = other.to_dict()

        # Remove excluded keys from dictionaries
        for key in exclude:
            self_dict.pop(key, None)
            other_dict.pop(key, None)

        # Now compare the modified dictionaries
        return self_dict == other_dict


class HFTextGenParams(_SharedParams):
    """Configuration params container for Langchain HuggingFaceTextGenInference wrapper.

    See docs here for details of available generation params:
        https://python.langchain.com/docs/integrations/llms/huggingface_textgen_inference

    """

    host: str = "localhost"
    max_new_tokens: int = 512
    top_k: Optional[int] = None
    top_p: Optional[float] = 0.95
    typical_p: Optional[float] = 0.95
    temperature: Optional[float] = 0.8
    repetition_penalty: Optional[float] = None
    return_full_text: bool = False
    truncate: Optional[int] = None
    stop_sequences: List[str] = Field(default_factory=list)
    seed: Optional[int] = None
    timeout: int = 120
    streaming: bool = False
    do_sample: bool = False
    watermark: bool = False

    @root_validator
    def _validate(cls, values):
        values["inference_server_url"] = f"http://{values['host']}:{values['port']}"
        return values

    def to_dict(self):
        """Get dict of valid args for HFTextGen instantiation.

        Since Langchain HuggingFaceTextGenInference only accepts an
        'inference_server_url` arg (not host/port like we're using) we
        need to override dict so it pops the host/port args and
        only include the inference server URL, which we derive
        from host/port.

        """
        return super().to_dict(exclude={"host", "port"})

    def compute_md5(self) -> str:
        """Compute MD5 hash based on parameter values."""
        params_dict = self.to_dict()
        params_str = json.dumps(
            params_dict, sort_keys=True
        )  # Serialize to JSON with sorted keys for consistency
        md5_hash = hashlib.md5(params_str.encode()).hexdigest()
        return md5_hash


class HuggingFaceTextGenInferenceAuto:
    HEALTH_CHECK_TIMEOUT = 60
    DOWNLOAD_TIMEOUT = (
        600  # 10 minutes; will probably have to raise this for larger models
    )
    DEFAULT_PORT = 8080  # Default **host** port for inference server
    AUTO_INCREMENT_PORT = True
    VALID_PORT_RANGE = (1024, 65535)
    MAX_PORT_INCREMENT_ATTEMPTS = 10
    DEFAULT_MODELS_CACHE_DIR = (
        Path("~").expanduser() / ".langchain-textgen" / "models"
    ).absolute()

    @staticmethod
    def _get_model_volume_path(container: Container) -> Tuple[str, str]:
        model_volume_path = container.attrs["Mounts"][0]
        host_path = model_volume_path["Source"]
        container_path = model_volume_path["Destination"]
        return host_path, container_path

    @staticmethod
    def _validate_cached_model(
        model_path_on_host: Path, model_path_on_container: DockerPath
    ):
        # This is an area for improvement if we wanted to do
        # some more validation, e.g. make sure the
        # model isn't corrupted, etc., or if we wanted
        # to allow options for the user to force re-downloading
        # For now we'll just make this a dummy method and
        # leave implementation to future work.

        # **** Improve in the future: add actual validation checks ****
        is_good_model = True
        if not is_good_model:
            raise ValueError(
                f"Cached model in {model_path_on_host} failed validation check."
            )

    @classmethod
    def _wait_for_model_download(cls, container: Container) -> None:
        """Wait for the model download to complete by monitoring container logs"""

        if container is None:
            raise ValueError("Existing container for model not found.")

        model_name = get_model_name_from_container(container)

        model_dir_name = hf_model_name_to_model_dir(model_name)
        (
            cached_model_path_on_host,
            cached_model_path_on_container,
        ) = cls._get_model_volume_path(container)
        cached_model_path_on_host = Path(cached_model_path_on_host) / model_dir_name
        cached_model_path_on_container = (
            DockerPath(container.short_id, cached_model_path_on_container)
            / model_dir_name
        )

        # Check if the model is already cached and that it matches on
        # host and container. Assumption is that we're bind mounting
        # models into our standard location in a hidden directory in
        # the users home directory, e.g. .cache., and that it is mounted
        # into the container at /data
        if str(os.path.join(*cached_model_path_on_container.parts[:2])) != "/data":
            raise ValueError(
                f"Models *must* be mounted in container to /data. "
                f"Got: {cached_model_path_on_container}"
            )

        if (
            cached_model_path_on_host.exists()
            and cached_model_path_on_container.exists()
        ):
            cls._validate_cached_model(
                cached_model_path_on_host, cached_model_path_on_container
            )
            model_size = bytes2human(get_directory_size(cached_model_path_on_host))
            print(
                f"Found cached model ({model_size}) - skipping download\n"
                f"\tHost Path:         {cached_model_path_on_host}\n"
                f"\tContainer Path:    {cached_model_path_on_container}"
            )
            return
        elif (
            not cached_model_path_on_host.exists()
            and cached_model_path_on_container.exists()
        ):
            # Edge cases, where user manually deletes a model
            # in the host cache while the model container is still running
            msg = (
                "Model volume cache mismatch. Model exists in container "
                "but doesn't exist on host. "
            )
            raise ValueError(msg)

        # Define regular expressions to match relevant log messages
        download_pattern = r"Downloaded .* in (\d+:\d+:\d+)."
        success_pattern = r"Successfully downloaded weights."

        with CountdownProgressBar(
            cls.DOWNLOAD_TIMEOUT,
            desc="Downloading model [0mb complete] (timeout countdown)",
        ) as pbar:
            # Keep checking the container logs until the download is complete
            start_time = time.time()
            download_complete = False

            while (
                not download_complete
                and time.time() - start_time < cls.DOWNLOAD_TIMEOUT
            ):
                logs = container.logs().decode("utf-8")

                # Search for patterns in the logs
                download_match = re.search(download_pattern, logs)
                success_match = re.search(success_pattern, logs)

                if download_match and success_match:
                    download_time = download_match.group(1)
                    model_size = bytes2human(
                        get_directory_size(cached_model_path_on_host)
                    )
                    print(
                        f"\nModel {model_name} download completed "
                        f"in {download_time} ({model_size})"
                    )
                    download_complete = True

                model_size = bytes2human(get_directory_size(cached_model_path_on_host))
                time.sleep(5)  # Wait for 5 seconds before checking again
                pbar.update(
                    5,
                    new_desc=f"Downloading model [{model_size} complete] "
                    f"(timeout countdown)",
                )

            if not download_complete:
                raise TimeoutError(f"Model download for '{model_name}' timed out.")

    @classmethod
    def _wait_for_status(cls, container, inference_server_url: str):
        """Wait for the container and inference API to become healthy"""

        # First handle waiting for download to complete, if it isn't cached
        cls._wait_for_model_download(container)

        is_container_ready = False
        is_api_ready = False
        with CountdownProgressBar(
            total=cls.HEALTH_CHECK_TIMEOUT,
            desc="Waiting for container/inference server to start",
        ) as pbar:
            start_time = time.time()
            while time.time() - start_time < cls.HEALTH_CHECK_TIMEOUT:
                msg = (
                    f"Waiting for container/inference server to start "
                    f"(CONTAINER={int(is_container_ready)}, "
                    f"API={int(is_api_ready)})"
                )

                if is_container_ready and is_api_ready:
                    print("\n***** Container and inference API are ready! *****\n")
                    break

                if container.status in ["running", "created"]:
                    is_container_ready = True

                    # Add a health check for the /health API
                    health_check_start_time = time.time()
                    health_check_url = f"{inference_server_url}/health"

                    while (
                        time.time() - health_check_start_time < cls.HEALTH_CHECK_TIMEOUT
                    ):
                        msg = (
                            f"Waiting for container/inference server to start "
                            f"(CONTAINER={int(is_container_ready)}, "
                            f"API={int(is_api_ready)})"
                        )

                        if is_api_ready:
                            break

                        try:
                            response = requests.get(health_check_url)
                            if response.status_code == 200:
                                is_api_ready = True
                        except requests.ConnectionError:
                            pass

                        time.sleep(1)
                        pbar.update(1, new_desc=msg)

                time.sleep(1)
                pbar.update(1, new_desc=msg)

            if is_container_ready and is_api_ready:
                return container
            else:
                # Future improvement: try to print out the logs to try to give the user
                # some indication of what happened here
                raise TimeoutError(
                    f"Container startup timeout. "
                    f"Did not receive healthy status within "
                    f"{cls.HEALTH_CHECK_TIMEOUT} seconds."
                    f"\tIs container ready:     {is_container_ready}",
                    f"\tIs inference API ready: {is_api_ready}",
                )

    @classmethod
    def _create_reuse_or_recreate_container(
        cls, client, docker_params: DockerParams, generation_params: HFTextGenParams
    ) -> Tuple[Container, DockerParams, HFTextGenParams]:
        """
        Create/recreate a container with the given parameters, or re-use existing one.

        Check if a container instantiated with the same parameters exists.
        If it exists, re-use it. If not, create a new one.

        In the event that everything *except* the model name matches, the original
        container will be shut down, and a new one recreated with the desired model.
        """
        print(
            "\n***** HuggingFaceTextGenInference "
            "Container/Server Startup Procedure *****"
        )

        # This is the desired state the user wants
        target_params = docker_params

        # First list all running containers for our service, and get
        # the params that were used to construct them
        containers_in_service = [
            c for c in client.containers.list() if c.name.startswith(SERVICE_NAME)
        ]
        for existing_container in containers_in_service:
            existing_params = DockerParams.from_container(existing_container)

            # If we find a container that matches what we want already,
            # we immediately return it
            if target_params.container_name == existing_params.container_name:
                print("Founding matching container that already exists. Re-using it.")
                return existing_container, docker_params, generation_params

            # If everything matches *except* the model the user wants to use,
            # then assume user wants to re-create the container using new model
            if target_params.equals_with_exclusions(
                existing_params, exclude="pretrained_model_name"
            ):
                existing_model_name = get_model_name_from_container(existing_container)
                print(
                    f"Container exists, but user wants to swap models: "
                    f"{existing_model_name} --> {docker_params.pretrained_model_name}\n"
                    f"Shutting down existing container; restarting with desired model."
                )
                cls.shutdown(container_name=existing_container.name, timeout=30)
                break

        # Otherwise, go ahead and create a new container with the updated parameters
        # First check if any of the specified params are going to run into conflicts
        # (e.g. host port availability). If the port isn't available and
        # AUTO_INCREMENT_PORT=True, try to find the next available port and update
        # the parameters accordingly.
        if not is_host_port_available(client, docker_params.port):
            print(f"Host port {docker_params.port} unavailable. ")
            if not cls.AUTO_INCREMENT_PORT:
                raise ValueError(
                    f"Host port {docker_params.port} unavailable and "
                    f"AUTO_INCREMENT_PORT={cls.AUTO_INCREMENT_PORT}"
                    f"Set AutoHuggingFaceTextGenInference.AUTO_INCREMENT_PORT=True "
                    f"to try to find available port"
                )
            new_port = find_available_host_port(
                client=client,
                desired_port=docker_params.port,
                auto_increment=cls.AUTO_INCREMENT_PORT,
                max_retries=cls.MAX_PORT_INCREMENT_ATTEMPTS,
                port_range=cls.VALID_PORT_RANGE,
            )
            docker_params.port = new_port
            generation_params.port = new_port

        d_params_docker = docker_params.to_dict()

        container = start_model_container(
            client=client,
            container_name=docker_params.container_name,
            **d_params_docker,
        )
        return container, docker_params, generation_params

    @classmethod
    def _from_params(
        cls,
        client,
        generation_params: HFTextGenParams,
        docker_params: DockerParams,
    ) -> HuggingFaceTextGenInference:
        # Check if a container with the same name and parameters exists,
        # and recreate if needed
        (
            container,
            docker_params,
            generation_params,
        ) = cls._create_reuse_or_recreate_container(
            client, docker_params, generation_params
        )

        # Wait for healthy status based on running container and inference API
        cls._wait_for_status(container, generation_params.inference_server_url)

        llm = HuggingFaceTextGenInference(**generation_params.to_dict())

        # Store docker params in server_kwargs key that Langchain provides.
        # These values might come in handy later.
        llm.server_kwargs["docker"] = docker_params.to_dict()

        return llm

    @classmethod
    @validate_environment
    def shutdown(cls, container_name: str, timeout=30) -> None:
        """Stop and removes a container using its ID, short ID, or container name

        Since we've hardcoded `remove=True` in the startup method, shutting down
        automatically removes the container (doing it this way gives up some
        flexibility, but makes state management a lot easier, which we really
        want to avoid doing from within our python code, and let docker handle
        as much container state management as possible).

        """
        client = docker.from_env()
        stop_container(client, container_name, timeout)

    @classmethod
    @validate_environment
    def get_docker_config_from_llm(cls, llm: HuggingFaceTextGenInference) -> dict:
        """Get dict of docker params used to instantiate the textgen-inference-server.

        Can easily recreate a new instance of the params this way:

            llm = ... AutoHuggingFaceTextGenInference.from_docker(...)
            d = AutoHuggingFaceTextGenInference.get_docker_config_from_llm(llm)
            p = DockerParams(**d)

        """
        return llm.server_kwargs.get("docker", {})

    @classmethod
    @validate_environment
    def get_container_name_from_llm(cls, llm: HuggingFaceTextGenInference) -> str:
        client = docker.from_env()
        cfg: dict = cls.get_docker_config_from_llm(llm)
        if cfg:
            d = DockerParams(**cfg)
            container = get_container(client, name=d.container_name)
            model_name = get_model_name_from_container(container)
            return model_name
        else:
            return ""

    @classmethod
    @validate_environment
    def from_docker(cls, model_name: str, **kwargs) -> HuggingFaceTextGenInference:
        """Automated inference server startup + HuggingFaceTextGenInference."""
        kwargs = {**{"pretrained_model_name": model_name}, **kwargs}
        kwargs["port"] = kwargs.pop("port", cls.DEFAULT_PORT)
        kwargs["volume_path"] = Path(
            kwargs.pop("volume_path", cls.DEFAULT_MODELS_CACHE_DIR)
        )

        client = docker.from_env()
        docker_params = DockerParams(**kwargs)
        generation_params = HFTextGenParams(**kwargs)

        return cls._from_params(client, generation_params, docker_params)
