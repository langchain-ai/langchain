import argparse
import json
import logging
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from subprocess import CalledProcessError
from typing import Dict, Generator, List, Mapping, Optional, Union, cast

import requests
import yaml

from langchain.env import get_runtime_environment

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_DIR = Path(__file__).parent


def pprint_services(services_status: List[Mapping[str, Union[str, List[str]]]]) -> None:
    # Loop through and collect Service, State, and Publishers["PublishedPorts"]
    # for each service
    services = []
    for service in services_status:
        service_status: Dict[str, str] = {
            "Service": str(service["Service"]),
            "Status": str(service["Status"]),
        }
        publishers = cast(List[Dict], service.get("Publishers", []))
        if publishers:
            service_status["PublishedPorts"] = ", ".join(
                [str(publisher["PublishedPort"]) for publisher in publishers]
            )
        services.append(service_status)

    max_service_len = max(len(service["Service"]) for service in services)
    max_state_len = max(len(service["Status"]) for service in services)
    service_message = [
        "\n"
        + "Service".ljust(max_service_len + 2)
        + "Status".ljust(max_state_len + 2)
        + "Published Ports"
    ]
    for service in services:
        service_str = service["Service"].ljust(max_service_len + 2)
        state_str = service["Status"].ljust(max_state_len + 2)
        ports_str = service.get("PublishedPorts", "")
        service_message.append(service_str + state_str + ports_str)

    langchain_endpoint: str = "http://localhost:1984"
    used_ngrok = any(["ngrok" in service["Service"] for service in services])
    if used_ngrok:
        langchain_endpoint = get_ngrok_url(auth_token=None)

    service_message.append(
        "\nTo connect, set the following environment variables"
        " in your LangChain application:"
        "\nLANGCHAIN_TRACING_V2=true"
        f"\nLANGCHAIN_ENDPOINT={langchain_endpoint}"
    )
    logger.info("\n".join(service_message))


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
        openai_api_key: Optional[str] = None,
    ) -> None:
        """Run the LangChainPlus server locally.

        Args:
            expose: If True, expose the server to the internet using ngrok.
            auth_token: The ngrok authtoken to use (visible in the ngrok dashboard).
                If not provided, ngrok server session length will be restricted.
            dev: If True, use the development (rc) image of LangChainPlus.
            openai_api_key: The OpenAI API key to use for LangChainPlus
                If not provided, the OpenAI API Key will be read from the
                OPENAI_API_KEY environment variable. If neither are provided,
                some features of LangChainPlus will not be available.
        """
        if dev:
            os.environ["_LANGCHAINPLUS_IMAGE_PREFIX"] = "rc-"
        if openai_api_key is not None:
            os.environ["OPENAI_API_KEY"] = openai_api_key
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

    def status(self) -> None:
        """Provide information about the status LangChainPlus server."""

        command = [
            *self.docker_compose_command,
            "-f",
            str(self.docker_compose_file),
            "ps",
            "--format",
            "json",
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            command_stdout = result.stdout.decode("utf-8")
            services_status = json.loads(command_stdout)
        except json.JSONDecodeError:
            logger.error("Error checking LangChainPlus server status.")
            return
        if services_status:
            logger.info("The LangChainPlus server is currently running.")
            pprint_services(services_status)
        else:
            logger.info("The LangChainPlus server is not running.")
            return


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
    server_start_parser.add_argument(
        "--openai-api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="The OpenAI API key to use for LangChainPlus."
        " If not provided, the OpenAI API Key will be read from the"
        " OPENAI_API_KEY environment variable. If neither are provided,"
        " some features of LangChainPlus will not be available.",
    )
    server_start_parser.set_defaults(
        func=lambda args: server_command.start(
            expose=args.expose,
            auth_token=args.ngrok_authtoken,
            dev=args.dev,
            openai_api_key=args.openai_api_key,
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
    server_status_parser = server_subparsers.add_parser(
        "status", description="Show the LangChainPlus server status."
    )
    server_status_parser.set_defaults(func=lambda args: server_command.status())
    env_parser = subparsers.add_parser("env")
    env_parser.set_defaults(func=lambda args: env())

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
