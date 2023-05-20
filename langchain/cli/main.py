import argparse
import logging
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from subprocess import CalledProcessError
from typing import Generator, List, Optional

import requests
import yaml

from langchain.env import get_runtime_environment

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_DIR = Path(__file__).parent


def get_docker_compose_command() -> List[str]:
    """Get the correct docker compose command for this system."""
    try:
        subprocess.check_call(
            ["docker", "compose", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return ["docker", "compose"]
    except (CalledProcessError, FileNotFoundError):
        try:
            subprocess.check_call(
                ["docker-compose", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return ["docker-compose"]
        except (CalledProcessError, FileNotFoundError):
            raise ValueError(
                "Neither 'docker compose' nor 'docker-compose'"
                " commands are available. Please install the Docker"
                " server following the instructions for your operating"
                " system at https://docs.docker.com/engine/install/"
            )


def get_ngrok_url(auth_token: Optional[str]) -> str:
    """Get the ngrok URL for the LangChainPlus server."""
    ngrok_url = "http://localhost:4040/api/tunnels"
    try:
        response = requests.get(ngrok_url)
        response.raise_for_status()
        exposed_url = response.json()["tunnels"][0]["public_url"]
    except requests.exceptions.HTTPError:
        raise ValueError("Could not connect to ngrok console.")
    except (KeyError, IndexError):
        message = "ngrok failed to start correctly. "
        if auth_token is not None:
            message += "Please check that your authtoken is correct."
        raise ValueError(message)
    return exposed_url


@contextmanager
def create_ngrok_config(
    auth_token: Optional[str] = None,
) -> Generator[Path, None, None]:
    """Create the ngrok configuration file."""
    config_path = _DIR / "ngrok_config.yaml"
    if config_path.exists():
        # If there was an error in a prior run, it's possible
        # Docker made this a directory instead of a file
        if config_path.is_dir():
            shutil.rmtree(config_path)
        else:
            config_path.unlink()
    ngrok_config = {
        "tunnels": {
            "langchain": {
                "proto": "http",
                "addr": "langchain-backend:8000",
            }
        },
        "version": "2",
        "region": "us",
    }
    if auth_token is not None:
        ngrok_config["authtoken"] = auth_token
    config_path = _DIR / "ngrok_config.yaml"
    with config_path.open("w") as f:
        yaml.dump(ngrok_config, f)
    yield config_path
    # Delete the config file after use
    config_path.unlink(missing_ok=True)


class PlusCommand:
    """Manage the LangChainPlus Tracing server."""

    def __init__(self) -> None:
        self.docker_compose_command = get_docker_compose_command()
        self.docker_compose_file = (
            Path(__file__).absolute().parent / "docker-compose.yaml"
        )
        self.ngrok_path = Path(__file__).absolute().parent / "docker-compose.ngrok.yaml"

    def _open_browser(self, url: str) -> None:
        try:
            subprocess.run(["open", url])
        except FileNotFoundError:
            pass

    def _start_local(self) -> None:
        command = [
            *self.docker_compose_command,
            "-f",
            str(self.docker_compose_file),
        ]
        subprocess.run(
            [
                *command,
                "up",
                "--pull=always",
                "--quiet-pull",
                "--wait",
            ]
        )
        logger.info(
            "langchain plus server is running at http://localhost.  To connect"
            " locally, set the following environment variable"
            " when running your LangChain application."
        )

        logger.info("\tLANGCHAIN_TRACING_V2=true")
        self._open_browser("http://localhost")

    def _start_and_expose(self, auth_token: Optional[str]) -> None:
        with create_ngrok_config(auth_token=auth_token):
            command = [
                *self.docker_compose_command,
                "-f",
                str(self.docker_compose_file),
                "-f",
                str(self.ngrok_path),
            ]
            subprocess.run(
                [
                    *command,
                    "up",
                    "--pull=always",
                    "--quiet-pull",
                    "--wait",
                ]
            )
        logger.info(
            "ngrok is running. You can view the dashboard at http://0.0.0.0:4040"
        )
        ngrok_url = get_ngrok_url(auth_token)
        logger.info(
            "langchain plus server is running at http://localhost."
            " To connect remotely, set the following environment"
            " variable when running your LangChain application."
        )
        logger.info("\tLANGCHAIN_TRACING_V2=true")
        logger.info(f"\tLANGCHAIN_ENDPOINT={ngrok_url}")
        self._open_browser("http://0.0.0.0:4040")
        self._open_browser("http://localhost")

    def start(
        self,
        *,
        expose: bool = False,
        auth_token: Optional[str] = None,
        dev: bool = False,
    ) -> None:
        """Run the LangChainPlus server locally.

        Args:
            expose: If True, expose the server to the internet using ngrok.
            auth_token: The ngrok authtoken to use (visible in the ngrok dashboard).
                If not provided, ngrok server session length will be restricted.
        """
        if dev:
            os.environ["_LANGCHAINPLUS_IMAGE_PREFIX"] = "rc-"
        if expose:
            self._start_and_expose(auth_token=auth_token)
        else:
            self._start_local()

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

    def logs(self) -> None:
        """Print the logs from the LangChainPlus server."""
        subprocess.run(
            [
                *self.docker_compose_command,
                "-f",
                str(self.docker_compose_file),
                "-f",
                str(self.ngrok_path),
                "logs",
            ]
        )


def env() -> None:
    """Print the runtime environment information."""
    env = get_runtime_environment()
    logger.info("LangChain Environment:")
    logger.info("\n".join(f"{k}:{v}" for k, v in env.items()))


def main() -> None:
    """Main entrypoint for the CLI."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(description="LangChainPlus CLI commands")

    server_command = PlusCommand()
    server_parser = subparsers.add_parser("plus", description=server_command.__doc__)
    server_subparsers = server_parser.add_subparsers()

    server_start_parser = server_subparsers.add_parser(
        "start", description="Start the LangChainPlus server."
    )
    server_start_parser.add_argument(
        "--expose",
        action="store_true",
        help="Expose the server to the internet using ngrok.",
    )
    server_start_parser.add_argument(
        "--ngrok-authtoken",
        default=os.getenv("NGROK_AUTHTOKEN"),
        help="The ngrok authtoken to use (visible in the ngrok dashboard)."
        " If not provided, ngrok server session length will be restricted.",
    )
    server_start_parser.add_argument(
        "--dev",
        action="store_true",
        help="Use the development version of the LangChainPlus image.",
    )
    server_start_parser.set_defaults(
        func=lambda args: server_command.start(
            expose=args.expose, auth_token=args.ngrok_authtoken, dev=args.dev
        )
    )

    server_stop_parser = server_subparsers.add_parser(
        "stop", description="Stop the LangChainPlus server."
    )
    server_stop_parser.set_defaults(func=lambda args: server_command.stop())

    server_logs_parser = server_subparsers.add_parser(
        "logs", description="Show the LangChainPlus server logs."
    )
    server_logs_parser.set_defaults(func=lambda args: server_command.logs())

    env_parser = subparsers.add_parser("env")
    env_parser.set_defaults(func=lambda args: env())

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
