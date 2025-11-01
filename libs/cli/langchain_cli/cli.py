"""LangChain CLI."""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Annotated

import typer

from langchain_cli._version import __version__
from langchain_cli.namespaces import app as app_namespace
from langchain_cli.namespaces import integration as integration_namespace
from langchain_cli.namespaces import template as template_namespace
from langchain_cli.namespaces.migrate import main as migrate_namespace
from langchain_cli.utils.packages import get_langserve_export, get_package_root

class LogLevel(str, Enum):
    critical = "CRITICAL"
    error = "ERROR"
    warning = "WARNING"
    info = "INFO"
    debug = "DEBUG"

def _configure_logging(level: LogLevel) -> None:
    logging.basicConfig(
        level=getattr(logging, level.value, logging.INFO),
        format="%(levelname)s | %(name)s | %(message)s",
    )

app = typer.Typer(no_args_is_help=True, add_completion=False)

app.add_typer(
    template_namespace.package_cli,
    name="template",
    help=template_namespace.__doc__,
)
app.add_typer(app_namespace.app_cli, name="app", help=app_namespace.__doc__)
app.add_typer(
    integration_namespace.integration_cli,
    name="integration",
    help=integration_namespace.__doc__,
)

app.command(
    name="migrate",
    context_settings={
        # Let Grit handle the arguments
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)(
    migrate_namespace.migrate,
)


def _version_callback(*, show_version: bool) -> None:
    if show_version:
        typer.echo(f"langchain-cli {__version__}")
        raise typer.Exit


@app.callback()
def _main(
    *,
    version: bool = typer.Option(
        False,  # noqa: FBT003
        "--version",
        "-v",
        help="Print the current CLI version.",
        callback=_version_callback,
        is_eager=True,
    ),
    log_level: LogLevel = typer.Option(
        LogLevel.info,
        "--log-level",
        help="Logging verbosity for this command (default: INFO).",
        show_default=False,
        case_sensitive=False,
    ),
) -> None:
    _configure_logging(log_level)


def _validate_port(port: int | None) -> int | None:
    if port is None:
        return None
    if not (0 < port < 65536):
        raise typer.BadParameter("Port must be between 1 and 65535.")
    return port


@app.command()
def serve(
    *,
    port: Annotated[
        int | None,
        typer.Option(help="The port to run the server on"),
    ] = None,
    host: Annotated[
        str | None,
        typer.Option(help="The host to run the server on"),
    ] = None,
    reload: Annotated[
        bool,
        typer.Option("--reload/--no-reload", help="Enable autoreload if supported."),
    ] = False,
) -> None:
    """Start the LangServe app, whether it's a template or an app."""
    if host is None:
        host = os.getenv("LANGCHAIN_CLI_HOST") or None
    if port is None:
        env_port = os.getenv("LANGCHAIN_CLI_PORT")
        if env_port and env_port.isdigit():
            port = int(env_port)

    port = _validate_port(port)

    try:
        project_dir = get_package_root()
        pyproject = project_dir / "pyproject.toml"
        get_langserve_export(pyproject)
    except (KeyError, FileNotFoundError):
        _serve_with_reload_passthrough(app_namespace.serve, host=host, port=port, reload=reload)
    else:
        _serve_with_reload_passthrough(template_namespace.serve, host=host, port=port, reload=reload)


def _serve_with_reload_passthrough(func, *, host: str | None, port: int | None, reload: bool) -> None:
    """
    Call a `serve` function, passing `reload` only if it is accepted.
    Keeps backward compatibility if the target signature doesn't support it.
    """
    try:
        return func(port=port, host=host, reload=reload)  # type: ignore[call-arg]
    except TypeError:
        return func(port=port, host=host)

if __name__ == "__main__":
    app()
