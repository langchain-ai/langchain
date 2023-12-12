"""
Develop integration packages for LangChain.
"""

import re
import shutil
import subprocess
from pathlib import Path

import typer
from typing_extensions import Annotated

from langchain_cli.utils.find_replace import replace_glob

integration_cli = typer.Typer(no_args_is_help=True, add_completion=False)


@integration_cli.command()
def new(
    name: Annotated[
        str,
        typer.Option(
            help="The name of the integration to create. "
            "Do not include `langchain-`.",
            prompt=True,
        ),
    ],
):
    """
    Creates a new integration package.

    Should be run from libs/partners
    """
    name = name.lower()

    if name.startswith("langchain-"):
        typer.echo("Name should not start with `langchain-`.")
        raise typer.Exit(code=1)

    destination_dir = Path.cwd() / name
    if destination_dir.exists():
        typer.echo(f"Folder {destination_dir} exists.")
        raise typer.Exit(code=1)

    # copy over template from ../integration_template
    project_template_dir = Path(__file__).parents[1] / "integration_template"
    shutil.copytree(project_template_dir, destination_dir, dirs_exist_ok=False)

    package_name = f"langchain-{name}"
    module_name = re.sub(
        r"[^a-zA-Z0-9_]",
        "_",
        package_name,
    )

    # folder movement
    package_dir = destination_dir / module_name
    shutil.move(destination_dir / "integration_template", package_dir)

    # replacements in files
    replacements = {
        "__package_name__": package_name,
        "__module_name__": module_name,
    }
    replace_glob(destination_dir, "**/*", replacements)

    # poetry install
    subprocess.run(["poetry", "install"], cwd=destination_dir)
