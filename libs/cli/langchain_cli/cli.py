from typing import Optional

import typer
from gritql import run
from typing_extensions import Annotated

from langchain_cli.namespaces import app as app_namespace
from langchain_cli.namespaces import integration as integration_namespace
from langchain_cli.namespaces import template as template_namespace
from langchain_cli.namespaces.migrate import main as migrate_namespace
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


@app.command(
    context_settings={
        # Let Grit handle the arguments
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
    # Grit embeds its own help
    add_help_option=False,
)
def migrate(ctx: typer.Context):
    final_code = run.apply_pattern(
        "langchain_all_migrations()",
        ctx.args,
        grit_dir=migrate_namespace.get_gritdir_path(),
    )

    raise typer.Exit(code=final_code)


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


if __name__ == "__main__":
    app()
