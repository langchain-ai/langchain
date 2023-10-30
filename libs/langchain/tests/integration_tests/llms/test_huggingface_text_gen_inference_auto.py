import contextlib
import os
import shutil
import socket
import tempfile
from pathlib import Path

import pytest

from langchain.llms.huggingface_text_gen_inference_auto import (
    DockerParams,
    DockerPath,
    HuggingFaceTextGenInference,
    HuggingFaceTextGenInferenceAuto,
    check_gpu,
    check_nvidia_docker,
    check_os,
    check_prerequisites,
    find_available_host_port,
    get_container,
    get_model_name_from_container,
    is_host_port_available,
    is_valid_hf_model_name_format,
)


def check_port_available(host: str, port: int) -> bool:
    """Returns True if port is available on host, False otherwise."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except socket.error:
            return False
        else:
            return True


@pytest.mark.requires("docker")
@pytest.fixture(scope="session", autouse=True)
def check_host_port_availability() -> None:
    # Should probably create some sort of pool of ports tests can use;
    # good enough for now to just check that what we need is available
    host = "localhost"
    needed_ports = [8080, 8081, 8082]
    for port in needed_ports:
        if not check_port_available(host, port):
            import warnings

            message = (
                f"Skipping tests. Port {port} is in use on the host. "
                f"Ensure it is free then run tests again. "
            )
            warnings.warn(message)
            pytest.skip(f"Skipping tests: {message}", allow_module_level=True)


@pytest.mark.requires("docker")
@pytest.fixture(scope="session", autouse=True)
def check_compatibility_before_tests() -> None:
    """
    Ensures test suite completes gracefully regardless of system environment.
    The code under test introduces some fairly harsh requirements on the
    system (e.g. Linux OS, Nvidia GPU availability, etc.), so to prevent
    blowing up the Langchain deployment pipeline, all the tests here
    are skipped if a compatible test environment isn't detected.
    :return:
    """
    is_compatible, message = check_prerequisites()
    if not is_compatible:
        pytest.skip(f"Skipping tests: {message}", allow_module_level=True)


@pytest.mark.requires("docker")
@pytest.fixture(scope="session", autouse=True)
def docker_teardown() -> None:
    yield
    # Teardown
    import docker

    client = docker.from_env()
    for container in client.containers.list():
        if container.name.startswith("text-generation-inference--"):
            print(f"Stopping container: {container.name}")
            container.stop()
            container.wait(timeout=30)  # Wait for the container to stop (optional)


@contextlib.contextmanager
def tolerant_tempdir():
    """
    Docker permissions issues when dealing with bind mounts are annoying.
    Not a huge deal, as its only as issue in testing when we're trying
    not to leave a footprint. For now, we'll rely on many cleanup
    of temp resources. Long term should fix the user/group profiles in
    container to match the host when creating the container.

    Also, we'll create the temporary directory on the users home path
    so we don't run into issues with the docker user potentially not
    having permission to write to /tmp on the host machine.
    """
    # Wrap everything in an enclosing "temporary" directory to at least
    # make manual cleanup easier, in the event we do get permissions errors.
    # Do not modify this path to anything you are not okay with deleting!
    user_accessible_temp_dir = (
        Path("~").expanduser() / ".tmp-langchain-integration-tests"
    )
    user_accessible_temp_dir.mkdir(exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir=user_accessible_temp_dir)
    try:
        yield temp_dir
    finally:
        try:
            shutil.rmtree(temp_dir)
            # Enforce hardcoded path to make it harder to delete important stuff
            path2 = Path("~").expanduser() / ".tmp-langchain-integration-tests"
            if str(user_accessible_temp_dir) == str(path2):
                shutil.rmtree(user_accessible_temp_dir)
            else:
                raise ValueError(
                    "Temp path is hardcoded for safety. "
                    "Modify path and assertion if you really mean to change it. "
                )
        except PermissionError:
            print(
                f"Encountered PermissionsError while closing temp_dir. "
                f"Will need to manually clean up {temp_dir}."
            )


@pytest.fixture
def temp_models_dir():
    with tolerant_tempdir() as temp_dir:
        old_models_dir = HuggingFaceTextGenInferenceAuto.DEFAULT_MODELS_CACHE_DIR
        HuggingFaceTextGenInferenceAuto.DEFAULT_MODELS_CACHE_DIR = temp_dir
        yield temp_dir
        HuggingFaceTextGenInferenceAuto.DEFAULT_MODELS_CACHE_DIR = old_models_dir


@pytest.fixture
def model_name() -> str:
    # HuggingFace model. Try to use something small; since we're really running
    # integration tests here want to avoid issues with latency / dependencies
    # on external service providers
    return "bigscience/bloom-560m"


@pytest.mark.requires("docker")
@pytest.fixture
def docker_client():
    import docker

    client = docker.from_env()
    yield client
    for container in client.containers.list():
        if container.name.startswith("text-generation-inference--"):
            container.stop()
            container.wait(timeout=30)


@pytest.mark.requires("docker")
@pytest.fixture(scope="session")
def docker_container():
    import docker

    client = docker.from_env()
    container = client.containers.run(
        "alpine", command="/bin/sh", detach=True, tty=True
    )
    yield container
    container.stop()
    container.wait(timeout=30)
    container.remove(force=True)


@pytest.mark.requires("docker")
@pytest.fixture
def setup_docker_containers_for_port_tests(docker_client):
    image_name = "alpine:3.18.4"
    docker_client.images.pull(image_name)
    container1 = container2 = None
    try:
        # Create and start some containers with known port bindings
        container1 = docker_client.containers.run(
            image_name, "sleep infinity", detach=True, ports={"80/tcp": 8080}
        )
        container2 = docker_client.containers.run(
            image_name, "sleep infinity", detach=True, ports={"80/tcp": 8081}
        )
        yield
    finally:
        # Stop and remove containers after tests
        if container1:
            container1.stop()
            container1.wait(timeout=30)
            container1.remove(force=True)
        if container2:
            container2.stop()
            container2.wait(timeout=30)
            container2.remove(force=True)


def test_is_valid_hf_model_name_format():
    # Test the function with a valid model name
    model_name = "model-owner/model-name"
    assert is_valid_hf_model_name_format(model_name)

    # Test the function with an invalid model name
    model_name = "not-a-valid-name"
    assert not is_valid_hf_model_name_format(model_name)


@pytest.mark.requires("docker")
@pytest.mark.slow
def test_is_port_available(docker_client, setup_docker_containers_for_port_tests):
    assert is_host_port_available(docker_client, 8080) is False
    assert is_host_port_available(docker_client, 8081) is False
    assert is_host_port_available(docker_client, 8082) is True


@pytest.mark.requires("docker")
@pytest.mark.slow
def test_find_available_port_no_auto_increment(
    docker_client, setup_docker_containers_for_port_tests
):
    assert find_available_host_port(docker_client, 8080, auto_increment=False) is None
    assert find_available_host_port(docker_client, 8082, auto_increment=False) == 8082


@pytest.mark.requires("docker")
@pytest.mark.slow
def test_find_available_port_auto_increment(
    docker_client, setup_docker_containers_for_port_tests
):
    assert find_available_host_port(docker_client, 8080, auto_increment=True) == 8082
    assert find_available_host_port(docker_client, 8081, auto_increment=True) == 8082


@pytest.mark.requires("docker")
@pytest.mark.slow
def test_find_available_port_max_retries(
    docker_client, setup_docker_containers_for_port_tests
):
    with pytest.raises(ValueError) as exc_info:
        find_available_host_port(
            docker_client, 8080, auto_increment=True, max_retries=1
        )
    assert (
        str(exc_info.value)
        == "No available port found within 1 retries or port range (1024, 65535)."
    )


@pytest.mark.requires("docker")
@pytest.mark.slow
def test_invalid_port_ranges(docker_client):
    # Port value too low
    with pytest.raises(ValueError) as exc_info:
        find_available_host_port(docker_client, 8080, port_range=(1023, 9000))
    assert str(exc_info.value).startswith(
        "Port range must be in range 1024-65535, inclusive."
    )

    # Port value too high
    with pytest.raises(ValueError) as exc_info:
        find_available_host_port(docker_client, 8080, port_range=(1024, 65536))
    assert str(exc_info.value).startswith(
        "Port range must be in range 1024-65535, inclusive."
    )


@pytest.mark.requires("docker")
def test_dockerpath_exists(docker_container):
    dpath = DockerPath(docker_container.id, "/etc/passwd")
    assert dpath.exists() is True

    dpath = DockerPath(docker_container.id, "/nonexistent")
    assert dpath.exists() is False


@pytest.mark.requires("docker")
def test_dockerpath_is_dir(docker_container):
    dpath = DockerPath(docker_container.id, "/etc")
    assert dpath.is_dir() is True

    dpath = DockerPath(docker_container.id, "/etc/passwd")
    assert dpath.is_dir() is False


@pytest.mark.requires("docker")
def test_dockerpath_is_file(docker_container):
    dpath = DockerPath(docker_container.id, "/etc/passwd")
    assert dpath.is_file() is True

    dpath = DockerPath(docker_container.id, "/etc")
    assert dpath.is_file() is False


@pytest.mark.requires("docker")
def test_dockerpath_stat(docker_container):
    dpath = DockerPath(docker_container.id, "/etc/passwd")
    stat_result = dpath.stat()
    assert isinstance(stat_result, os.stat_result)
    assert isinstance(stat_result.st_size, int)
    assert stat_result.st_size > 0


@pytest.mark.requires("docker")
def test_dockerpath_concatenation(docker_container):
    dpath = DockerPath(docker_container.id, "/home/user")
    concat_path = dpath / "some_dir"
    assert str(concat_path) == "/home/user/some_dir"
    assert hasattr(concat_path, "container_id")


@pytest.mark.requires("docker")
def test_check_os():
    is_compatible, message = check_os()
    # The exact assertions here may depend on the system the tests are being run on
    assert isinstance(is_compatible, bool)
    assert isinstance(message, str)


@pytest.mark.requires("docker")
def test_check_gpu():
    is_gpu_available = check_gpu()
    # The exact assertion here may depend on the system the tests are being run on
    assert isinstance(is_gpu_available, bool)


@pytest.mark.requires("docker")
def test_check_nvidia_docker():
    is_nvidia_docker_available = check_nvidia_docker()
    # The exact assertion here may depend on the system the tests are being run on
    assert isinstance(is_nvidia_docker_available, bool)


@pytest.mark.requires("docker")
def test_check_prerequisites():
    is_compatible, message = check_prerequisites()
    # The exact assertions here may depend on the system the tests are being run on
    assert isinstance(is_compatible, bool)
    assert isinstance(message, str)


@pytest.mark.requires("docker")
@pytest.mark.slow
def test_startup_shutdown(docker_client, temp_models_dir, model_name):
    # Test starting and stopping a container
    llm = HuggingFaceTextGenInferenceAuto.from_docker(
        model_name, host="0.0.0.0", port=8081, shm_size="1g"
    )
    assert isinstance(llm, HuggingFaceTextGenInference)

    # Ensure the container is running
    d = DockerParams(**HuggingFaceTextGenInferenceAuto.get_docker_config_from_llm(llm))
    container = get_container(docker_client, name=d.container_name)
    assert container.status == "running"

    # Shutdown the container
    HuggingFaceTextGenInferenceAuto.shutdown(container_name=d.container_name)
    container = get_container(docker_client, name=d.container_name)
    assert container is None  # Container should no longer exist


@pytest.mark.requires("docker")
@pytest.mark.slow
def test_container_model_name_swap(docker_client, temp_models_dir, model_name):
    # Start the container initially
    llm = HuggingFaceTextGenInferenceAuto.from_docker(
        model_name, host="0.0.0.0", port=8081, shm_size="1g"
    )
    d = DockerParams(**HuggingFaceTextGenInferenceAuto.get_docker_config_from_llm(llm))
    container = get_container(docker_client, name=d.container_name)
    model1_name = get_model_name_from_container(container)
    assert container.status == "running"

    # Create the container with a different model
    new_model_name = "ComCom/gpt2-small"
    llm_new = HuggingFaceTextGenInferenceAuto.from_docker(
        new_model_name, host="0.0.0.0", port=8081, shm_size="1g"
    )
    d_new = DockerParams(
        **HuggingFaceTextGenInferenceAuto.get_docker_config_from_llm(llm_new)
    )
    container_new = get_container(docker_client, name=d_new.container_name)
    model2_name = get_model_name_from_container(container_new)
    assert container_new.status == "running"

    # Doing the model swap should have shut down existing container, and
    # recreated with new model running on the same port as the previous.
    # Any other changes would have generated an auto-increment to the port,
    # so check that same port is still in use as a basic sanity test
    assert not check_port_available("0.0.0.0", 8081)

    # Check that we got the model names we expected
    assert model1_name != model2_name

    # Shutdown the container
    HuggingFaceTextGenInferenceAuto.shutdown(d_new.container_name)
    container = get_container(docker_client, d_new.container_name)
    assert container is None  # Container should no longer exist


@pytest.mark.requires("docker")
def test_invalid_environment(docker_client, temp_models_dir):
    # Make sure we raise an exception if the python docker library is not installed
    import langchain

    langchain.llms.huggingface_text_gen_inference_auto.docker = None

    # Test invalid environment without the docker library installed
    # Do this for any methods intended to be exposed for the class
    # as a public interface.
    with pytest.raises(EnvironmentError):
        HuggingFaceTextGenInferenceAuto.from_docker(model_name="xyz/my-model")

    with pytest.raises(EnvironmentError):
        HuggingFaceTextGenInferenceAuto.shutdown(container_name="some_container")

    with pytest.raises(EnvironmentError):
        HuggingFaceTextGenInferenceAuto.get_docker_config_from_llm(llm=None)

    # Re-import to go back to original state
    import docker

    langchain.llms.huggingface_text_gen_inference_auto.docker = docker


@pytest.mark.requires("docker")
@pytest.mark.slow
def test_health_check_timeout(temp_models_dir, model_name):
    # Test health check timeout
    with pytest.raises(TimeoutError):
        HuggingFaceTextGenInferenceAuto.HEALTH_CHECK_TIMEOUT = 0
        HuggingFaceTextGenInferenceAuto.from_docker(
            model_name, host="0.0.0.0", port=8081, shm_size="1g"
        )
