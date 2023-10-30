from typing import Optional

import typer
from typing_extensions import Annotated

from langchain_cli.namespaces import app_cli
from langchain_cli.namespaces import package_cli

app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(package_cli.hub, name="package", help=package_cli.__doc__)
app.add_typer(app_cli.app_cli, name="app", help=app_cli.__doc__)


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
        package_cli.start(port=port, host=host)
    except KeyError:
        app_cli.start(port=port, host=host)


if __name__ == "__main__":
    app()
