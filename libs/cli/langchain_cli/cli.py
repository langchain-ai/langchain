from typing import Optional

import typer
from typing_extensions import Annotated

from langchain_cli.namespaces import app as app_namespace
from langchain_cli.namespaces import integration as integration_namespace
from langchain_cli.namespaces import template as template_namespace
from langchain_cli.utils.packages import get_langserve_export, get_package_root

__version__ = "0.0.22rc0"

app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(
    template_namespace.package_cli, name="template", help=template_namespace.__doc__
)
app.add_typer(app_namespace.app_cli, name="app", help=app_namespace.__doc__)
app.add_typer(
    integration_namespace.integration_cli,
    name="integration",
    help=integration_namespace.__doc__,
)


def version_callback(show_version: bool) -> None:
    if show_version:
        typer.echo(f"gigachain-cli {__version__}")
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
    Start the GigaServe app, whether it's a template or an app.
    """

    # see if is a template
    try:
        project_dir = get_package_root()
        pyproject = project_dir / "pyproject.toml"
        get_langserve_export(pyproject)
    except KeyError:
        # not a template
        app_namespace.serve(port=port, host=host)
    else:
        # is a template
        template_namespace.serve(port=port, host=host)


@app.command()
def install_rus_certs() -> None:
    """
    Установка корневого сертификата Минцифры.
    """
    import certifi
    import httpx

    url = "https://gu-st.ru/content/Other/doc/russian_trusted_root_ca.cer"

    with open(certifi.where()) as infile:
        installed = url in infile.read()

    if installed:
        exit("ERR: Сертификат Минцифры уже установлен!")
    else:
        response = httpx.get(url, verify=False)
        response.raise_for_status()
        with open(certifi.where(), "ba") as outfile:
            outfile.write(b"\n\n# --- BEGIN %s ---\n\n" % url.encode())
            outfile.write(response.content.replace(b"\r\n", b"\n"))
            outfile.write(b"\n\n# --- END %s ---\n\n" % url.encode())
        exit("Сертификат Минцифры установлен!")


if __name__ == "__main__":
    app()
