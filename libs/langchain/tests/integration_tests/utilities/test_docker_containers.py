from pathlib import Path
from typing import cast

import pytest
from docker_containers.docker_containers import (
    DockerContainer,
    DockerImage,
    generate_langchain_container_tag,
    get_docker_client,
)


def test_generate_langchain_container_tag() -> None:
    tag = generate_langchain_container_tag()
    assert tag.startswith("langchain")
    assert len(tag) > len("langchain")
    new_tag = generate_langchain_container_tag()
    assert tag != new_tag, "Tags should be different"


@pytest.mark.requires("docker")
def test_docker_image_throws_for_bad_name() -> None:
    with pytest.raises(ValueError):
        DockerImage(name="docker_image_which_should_not_exist_42")


@pytest.mark.requires("docker")
def run_container_cowsay(image: DockerImage) -> None:
    """Helper for testing - runs cowsay command and verifies it works."""
    # note that our `cowsay` adds moo prefix as commands are executed
    # by ENTRYPOINT defined in dockerfile.
    try:
        container = DockerContainer(image)
        ret_code, log = container.spawn_run("I like langchain!")
        assert ret_code == 0
        assert (
            log.find(b"moo I like langchain") >= 0
        ), "Cowsay should say same words with moo"
    finally:
        docker_client = get_docker_client()
        docker_client.images.remove(image.name)


@pytest.mark.requires("docker")
def test_build_image_from_dockerfile() -> None:
    dockerfile_path = Path(__file__).parent / "docker_test_data/Dockerfile"
    image = DockerImage.from_dockerfile(dockerfile_path, name="cow")
    run_container_cowsay(image)


@pytest.mark.requires("docker")
def test_build_image_from_dockerfile_dirpath() -> None:
    dockerfile_dir = Path(__file__).parent / "docker_test_data/"
    image = DockerImage.from_dockerfile(dockerfile_dir)
    run_container_cowsay(image)


@pytest.mark.requires("docker")
def test_docker_spawn_run_works() -> None:
    container = DockerContainer(DockerImage.from_tag("alpine"))
    status_code, logs = container.spawn_run(["echo", "hello", "world"])
    assert status_code == 0
    assert logs.find(b"hello world") >= 0

    status_code, logs = container.spawn_run("echo good bye")
    assert status_code == 0
    assert logs.find(b"good bye") >= 0


@pytest.mark.requires("docker")
def test_docker_spawn_run_return_nonzero_status_code() -> None:
    container = DockerContainer(DockerImage.from_tag("alpine"))
    status_code, logs = container.spawn_run("sh -c 'echo hey && exit 1'")
    assert status_code == 1
    assert logs.find(b"hey") >= 0


@pytest.mark.requires("docker")
def test_docker_container_background_run_works() -> None:
    client = get_docker_client()
    container_name: str
    with DockerContainer(DockerImage.from_tag("alpine")) as container:
        container_name = container.name
        assert len(client.containers.list(filters={"name": container_name})) == 1
        ret_code, output = container.run("touch /animal.txt")
        assert ret_code == 0

        ret_code, output = container.run("ls /")
        assert ret_code == 0
        assert cast(bytes, output).find(b"animal.txt") >= 0

    assert len(client.containers.list(filters={"name": container_name})) == 0
