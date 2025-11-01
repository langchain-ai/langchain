"""LangChain CLI."""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Annotated, Any, Callable

import typer

from langchain_cli._version import __version__
from langchain_cli.namespaces import app as app_namespace
from langchain_cli.namespaces import integration as integration_namespace
from langchain_cli.namespaces import template as template_namespace
from langchain_cli.namespaces.migrate import main as migrate_namespace
from langchain_cli.utils.packages import get_langserve_export, get_package_root

PORT_MIN = 1
PORT_MAX = 65_535
MSG_INVALID_PORT = "Port must be between 1 and 65_535."

VERSION_OPT = typer.Option(  # noqa: B008 (allowed at module level)
    False,  # noqa: FBT003
    "--version",
    "-v",
    help="Print the current CLI version.",
    callback=lambda show: _version_callback(show_version=show),  # type: ignore[misc]
    is_eager=True,
)

PORT_OPT: Annotated[
    int | None, typer.Option
] = typer.Option(  # noqa: B008
    None,
    help="The port to run the server on.",
)

HOST_OPT: Annotated[
    str | None, typer.Option
] = typer.Option(
    None,
    help="The host to run the server on.",
)

RELOAD_OPT: Annotated[
    bool, typer.Option
] = typer.Option(
    False,
    "--reload/--no-reload",
    help="Enable autoreload if supported.",
)


class LogLevel(str, Enum):
    """Logging levels supported by the CLI."""

    critical = "CRITICAL"
    error = "ERROR"
    warning = "WARNING"
    info = "INFO"
    debug = "DEBUG"


def _configure_logging(level: LogLevel) -> None:
    """Configure root logging for CLI commands."""
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
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)(
    migrate_namespace.migrate,
)


def _version_callback(*, show_version: bool) -> None:
    """Echo the CLI version and exit when --version is provided."""
    if show_version:
        typer.echo(f"langchain-cli {__version__}")
        raise typer.Exit


@app.callback()
def _main(
    *,
    _version: bool = VERSION_OPT,
    log_level: LogLevel = typer.Option(
        LogLevel.info,
        "--log-level",
        help="Logging verbosity for this command (default: INFO).",
        show_default=False,
        case_sensitive=False,
    ),
) -> None:
    """Top-level callback configuring global options for the CLI."""
    _configure_logging(log_level)


def _validate_port(port: int | None) -> int | None:
    """Validate an optional TCP port value."""
    if port is None:
        return None
    if not (PORT_MIN <= port <= PORT_MAX):
        raise typer.BadParameter(MSG_INVALID_PORT)
    return port


@app.command()
def serve(
    *,
    port: Annotated[int | None, typer.Option] = PORT_OPT,
    host: Annotated[str | None, typer.Option] = HOST_OPT,
    reload: Annotated[bool, typer.Option] = RELOAD_OPT,  # noqa: A002
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
        _serve_with_reload_passthrough(
            app_namespace.serve, host=host, port=port, reload=reload
        )
    else:
        _serve_with_reload_passthrough(
            template_namespace.serve, host=host, port=port, reload=reload
        )


def _serve_with_reload_passthrough(
    func: Callable[..., Any],
    *,
    host: str | None,
    port: int | None,
    reload: bool,
) -> None:
    """Call a serve() function, passing `reload` only if it is accepted.

    This preserves backward compatibility with implementations that do not
    support a `reload` parameter.
    """
    try:
        func(port=port, host=host, reload=reload)
    except TypeError:
        func(port=port, host=host)


if __name__ == "__main__":
    app()
