"""Test the docker wrapper utility."""

import pytest
from langchain.utilities.docker import DockerWrapper, gvisor_runtime_available
from unittest.mock import MagicMock
import subprocess


def docker_installed() -> bool:
    """Checks if docker is installed locally."""
    try:
        subprocess.run(['which', 'docker',], check=True)
    except subprocess.CalledProcessError:
        return False

    return True




@pytest.mark.skipif(not docker_installed(), reason="docker not installed")
class TestDockerUtility:

    def test_command_default_image(self) -> None:
        """Test running a command with the default alpine image."""
        docker = DockerWrapper()
        output = docker.run('cat /etc/os-release')
        assert output.find(b'alpine')

    def test_inner_failing_command(self) -> None:
        """Test inner command with non zero exit"""
        docker = DockerWrapper()
        output = docker.run('ls /inner-failing-command')
        assert str(output).startswith("STDERR")

    def test_entrypoint_failure(self) -> None:
        """Test inner command with non zero exit"""
        docker = DockerWrapper()
        output = docker.run('todo handle APIError')
        assert output == 'ERROR'

    def test_check_gvisor_runtime(self) -> None:
        """test gVisor runtime verification using a mock docker client"""
        mock_client = MagicMock()
        mock_client.info.return_value = {'Runtimes': {'runsc': {'path': 'runsc'}}}
        assert gvisor_runtime_available(mock_client)
        mock_client.info.return_value = {'Runtimes': {'runc': {'path': 'runc'}}}
        assert not gvisor_runtime_available(mock_client)
