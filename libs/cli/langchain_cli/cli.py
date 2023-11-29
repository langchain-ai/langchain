from typing import Optional

import typer
from typing_extensions import Annotated

from langchain_cli.namespaces import hub, serve

app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(hub.hub, name="hub", help=hub.__doc__)
app.add_typer(serve.serve, name="serve", help=serve.__doc__)


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
        hub.start(port=port, host=host)
    except KeyError:
        serve.start(port=port, host=host)


if __name__ == "__main__":
    app()
