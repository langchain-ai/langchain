import argparse
import shutil
import subprocess
from pathlib import Path

from langchain.env import get_runtime_environment


def run_server() -> None:
    """Run the langchain server locally."""
    p = Path(__file__).absolute().parent / "docker-compose.yaml"

    if shutil.which("docker-compose") is None:
        docker_compose_command = ["docker", "compose"]
    else:
        docker_compose_command = ["docker-compose"]

    subprocess.run(
        [
            *docker_compose_command,
            "-f",
            str(p),
            "up",
            "--pull=always",
            "--quiet-pull",
            "-d",
            "--wait",
        ]
    )
    # Open the browser to the server
    subprocess.run(["open", "http://localhost"])


def env() -> None:
    """Print the environment."""
    env = get_runtime_environment()
    print("LangChain Environment:")
    print("\n".join(f"{k}:{v}" for k, v in env.items()))


def main() -> None:
    """Main entrypoint for the CLI."""
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    server_parser = subparsers.add_parser("server")
    server_parser.set_defaults(func=run_server)

    env_parser = subparsers.add_parser("env")
    env_parser.set_defaults(func=env)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func()


if __name__ == "__main__":
    main()
