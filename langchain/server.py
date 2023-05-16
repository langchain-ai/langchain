"""Script to run langchain-server locally using Docker."""
import subprocess
from pathlib import Path
from subprocess import CalledProcessError


def check_command(command: str) -> bool:
    """Check if a command is available."""
    try:
        subprocess.run(
            command.split(" "),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except (FileNotFoundError, CalledProcessError):
        return False

def main() -> None:
    """Run the langchain server locally."""
    p = Path(__file__).absolute().parent / "docker-compose.yaml"

    if check_command('docker compose'):
        subprocess.run(["docker", "compose", "-f", str(p), "pull"])
        subprocess.run(["docker", "compose", "-f", str(p), "up"])
    elif check_command('docker-compose'):
        print("We recommend upgrading to Docker Compose V2." + \
            "'docker-compose' is being used as fallback.")
        subprocess.run(["docker-compose", "-f", str(p), "pull"])
        subprocess.run(["docker-compose", "-f", str(p), "up"])
    else:
        raise ValueError("`docker compose` or `docker-compose` is required to run" + \
                        " langchain-server.")

if __name__ == "__main__":
    main()
