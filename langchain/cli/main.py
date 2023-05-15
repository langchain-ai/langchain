import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List

import requests

from langchain.env import get_runtime_environment


def get_docker_compose_command() -> List[str]:
    if shutil.which("docker-compose") is None:
        return ["docker", "compose"]
    else:
        return ["docker-compose"]


def get_ngrok_url() -> str:
    """Get the Ngrok URL for the LangChainPlus server."""
    ngrok_url = "http://localhost:4040/api/tunnels"
    response = requests.get(ngrok_url)
    response.raise_for_status()
    return response.json()["tunnels"][0]["public_url"]


class ServerCommand:
    """Manage the LangChainPlus Tracing server."""

    def __init__(self) -> None:
        self.docker_compose_command = get_docker_compose_command()
        self.docker_compose_file = (
            Path(__file__).absolute().parent / "docker-compose.yaml"
        )
        self.ngrok_path = Path(__file__).absolute().parent / "docker-compose.ngrok.yaml"

    def start(self, use_ngrok: bool = False) -> None:
        """Run the LangChainPlus server locally.

        Args:
            use_ngrok: If True, expose the server to the internet using Ngrok.
        """
        command = [
            *self.docker_compose_command,
            "-f",
            str(self.docker_compose_file),
        ]
        if use_ngrok:
            command += ["-f", str(self.ngrok_path)]
        subprocess.run(
            [
                *command,
                "up",
                "--pull=always",
                "--quiet-pull",
                "--wait",
            ]
        )
        if use_ngrok:
            print("NGrok is running. You can view the dashboard at http://0.0.0.0:4040")
            ngrok_url = get_ngrok_url()
            print(
                "LangChain server is running at http://localhost."
                " To connect remotely, set the following environment variable when running your LangChain application."
            )
            print("LANGCHAIN_TRACING_V2=true")
            print(f"LANGCHAIN_ENDPOINT={ngrok_url}")
            subprocess.run(["open", "http://localhost"])
        else:
            print(
                "LangChain server is running at http://localhost.  To connect"
                " locally, set the following environment variable"
                " when running your LangChain application."
            )

            print("LANGCHAIN_TRACING_V2=true")
            subprocess.run(["open", "http://localhost"])

    def stop(self) -> None:
        """Stop the LangChainPlus server."""
        subprocess.run(
            [
                *self.docker_compose_command,
                "-f",
                str(self.docker_compose_file),
                "-f",
                str(self.ngrok_path),
                "down",
            ]
        )


def env() -> None:
    """Print the runtime environment information."""
    env = get_runtime_environment()
    print("LangChain Environment:")
    print("\n".join(f"{k}:{v}" for k, v in env.items()))


def main() -> None:
    """Main entrypoint for the CLI."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(description="LangChainPlus CLI commands")

    server_command = ServerCommand()
    server_parser = subparsers.add_parser("server", description=server_command.__doc__)
    server_subparsers = server_parser.add_subparsers()

    server_start_parser = server_subparsers.add_parser(
        "start", description="Start the LangChainPlus server."
    )
    server_start_parser.add_argument(
        "--use-ngrok",
        action="store_true",
        help="Expose the server to the internet using Ngrok.",
    )
    server_start_parser.set_defaults(
        func=lambda args: server_command.start(args.use_ngrok)
    )

    server_stop_parser = server_subparsers.add_parser(
        "stop", description="Stop the LangChainPlus server."
    )
    server_stop_parser.set_defaults(func=lambda args: server_command.stop())

    env_parser = subparsers.add_parser("env")
    env_parser.set_defaults(func=lambda args: env())

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
