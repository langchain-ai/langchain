from typing import Optional

import typer
from typing_extensions import Annotated

from langchain_cli.namespaces import app as app_namespace
from langchain_cli.namespaces import template as template_namespace

__version__ = "0.0.11"

app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(
    template_namespace.package_cli, name="template", help=template_namespace.__doc__
)
app.add_typer(app_namespace.app_cli, name="app", help=app_namespace.__doc__)


def version_callback(show_version: bool) -> None:
    if show_version:
        typer.echo(f"langchain-cli {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Print the current CLI version.",
        callback=version_callback,
        is_eager=True,
    ),
):
    pass


@app.command()
def serve(
    *,
    port: Annotated[
        Optional[int], typer.Option(help="The port to run the server on")
    ] = None,
    host: Annotated[
        Optional[str], typer.Option(help="The host to run the server on")
    ] = None,
) -> None:
    """
    Start the LangServe app, whether it's a template or an app.
    """

    # try starting template package, if error, try langserve
    try:
        template_namespace.serve(port=port, host=host)
    except KeyError:
        app_namespace.serve(port=port, host=host)


if __name__ == "__main__":
    app()
