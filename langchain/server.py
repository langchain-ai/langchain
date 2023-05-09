"""Script to run langchain-server locally using docker-compose."""
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    """Run the langchain server locally."""
    p = Path(__file__).absolute().parent / "docker-compose.yaml"

    if shutil.which("docker-compose") is None:
        docker_compose_command = ["docker", "compose"]
    else:
        docker_compose_command = ["docker-compose"]

    subprocess.run([*docker_compose_command, "-f", str(p), "pull"])
    subprocess.run([*docker_compose_command, "-f", str(p), "up"])


if __name__ == "__main__":
    main()
