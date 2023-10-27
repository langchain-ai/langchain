"""
Manage installable hub packages.
"""

import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

hub = typer.Typer(no_args_is_help=True, add_completion=False)


@hub.command()
def new(
    name: Annotated[str, typer.Argument(help="The name of the folder to create")],
    with_poetry: Annotated[
        bool,
        typer.Option(
            "--with-poetry/--no-poetry", help="Don't run poetry install"
        ),
    ] = False,
):
    """
    Creates a new hub package.
    """
    computed_name = name if name != "." else Path.cwd().name
    destination_dir = Path.cwd() / name if name != "." else Path.cwd()

    # copy over template from ../package_template
    project_template_dir = Path(__file__).parents[1] / "package_template"
    shutil.copytree(project_template_dir, destination_dir, dirs_exist_ok=name == ".")

    package_name_split = computed_name.split("/")
    package_name_last = (
        package_name_split[-2]
        if len(package_name_split) > 1 and package_name_split[-1] == ""
        else package_name_split[-1]
    )
    default_package_name = re.sub(
        r"[^a-zA-Z0-9_]",
        "_",
        package_name_last,
    )

    # replace template strings
    pyproject = destination_dir / "pyproject.toml"
    pyproject_contents = pyproject.read_text()
    pyproject.write_text(
        pyproject_contents.replace("__package_name__", default_package_name)
    )

    # move module folder
    package_dir = destination_dir / default_package_name
    shutil.move(destination_dir / "package_template", package_dir)

    # replace readme
    readme = destination_dir / "README.md"
    readme_contents = readme.read_text()
    readme.write_text(
        readme_contents.replace("__package_name_last__", package_name_last)
    )

    # poetry install
    if with_poetry:
        subprocess.run(["poetry", "install"], cwd=destination_dir)


@hub.command()
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
    Starts a demo LangServe instance for this hub package.
    """
    cmd = ["poetry", "run", "poe", "start"]
    if port is not None:
        cmd += ["--port", str(port)]
    if host is not None:
        cmd += ["--host", host]
    subprocess.run(cmd)
