"""Script to run langchain-server locally using docker-compose."""
import subprocess
from pathlib import Path

from langchainplus_sdk.cli.main import get_docker_compose_command


def main() -> None:
    """Run the langchain server locally."""
    p = Path(__file__).absolute().parent / "docker-compose.yaml"

    docker_compose_command = get_docker_compose_command()
    subprocess.run([*docker_compose_command, "-f", str(p), "pull"])
    subprocess.run([*docker_compose_command, "-f", str(p), "up"])


if __name__ == "__main__":
    main()
