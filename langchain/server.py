"""Script to run langchain-server locally using docker-compose."""
import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> bool:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", action="store_true")
    return parser.parse_args().v2


def main() -> None:
    """Run the langchain server locally."""
    is_v2 = parse_args()
    filename = "docker-compose-v2.yaml" if is_v2 else "docker-compose.yaml"
    p = Path(__file__).absolute().parent / filename
    print(p)

    if shutil.which("docker-compose") is None:
        docker_compose_command = ["docker", "compose"]
    else:
        docker_compose_command = ["docker-compose"]

    subprocess.run([*docker_compose_command, "-f", str(p), "pull"])
    subprocess.run([*docker_compose_command, "-f", str(p), "up"])


if __name__ == "__main__":
    main()
