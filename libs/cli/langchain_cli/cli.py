from typing import Optional

import typer
from typing_extensions import Annotated

from langchain_cli.namespaces import app as app_namespace
from langchain_cli.namespaces import package as package_namespace

app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(
    package_namespace.package_cli, name="package", help=package_namespace.__doc__
)
app.add_typer(app_namespace.app_cli, name="app", help=app_namespace.__doc__)


@app.command()
def start(
    *,
    port: Annotated[
        Optional[int], typer.Option(help="The port to run the server on")
    ] = None,
    host: Annotated[
        Optional[str], typer.Option(help="The host to run the server on")
    ] = None,
) -> None:
    """
    Start the LangServe instance, whether it's a hub package or a serve project.
    """

    # try starting hub package, if error, try langserve
    try:
        package_namespace.start(port=port, host=host)
    except KeyError:
        app_namespace.start(port=port, host=host)


if __name__ == "__main__":
    app()
