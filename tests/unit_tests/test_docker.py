"""Test the docker wrapper utility."""

import pytest
from langchain.utilities.docker import DockerWrapper


def test_command_default_image() -> None:
    """Test running a command with the default alpine image."""
    docker = DockerWrapper()
    output = docker.run("cat /etc/os-release")
    assert output.find(b"alpine")

def test_inner_failing_command() -> None:
    """Test inner command with non zero exit"""
    docker = DockerWrapper()
    output = docker.run("ls /inner-failing-command")
    assert str(output).startswith("STDERR")

def test_entrypoint_failure() -> None:
    """Test inner command with non zero exit"""
    docker = DockerWrapper()
    output = docker.run("todo handle APIError")



