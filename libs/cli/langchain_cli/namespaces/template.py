"""
Develop installable templates.
"""

import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import typer
from langserve.packages import get_langserve_export
from typing_extensions import Annotated

from langchain_cli.utils.packages import get_package_root

package_cli = typer.Typer(no_args_is_help=True, add_completion=False)


@package_cli.command()
def new(
    name: Annotated[str, typer.Argument(help="The name of the folder to create")],
    with_poetry: Annotated[
        bool,
        typer.Option("--with-poetry/--no-poetry", help="Don't run poetry install"),
    ] = False,
):
    """
    Creates a new template package.
    """
    computed_name = name if name != "." else Path.cwd().name
    destination_dir = Path.cwd() / name if name != "." else Path.cwd()

    # copy over template from ../package_template
    project_template_dir = Path(__file__).parents[1] / "package_template"
    shutil.copytree(project_template_dir, destination_dir, dirs_exist_ok=name == ".")

    package_name_split = computed_name.split("/")
    package_name = (
        package_name_split[-2]
        if len(package_name_split) > 1 and package_name_split[-1] == ""
        else package_name_split[-1]
    )
    module_name = re.sub(
        r"[^a-zA-Z0-9_]",
        "_",
        package_name,
    )

    # generate app route code
    chain_name = f"{module_name}_chain"
    app_route_code = (
        f"from {module_name} import chain as {chain_name}\n\n"
        f'add_routes(app, {chain_name}, path="/{package_name}")'
    )

    # replace template strings
    pyproject = destination_dir / "pyproject.toml"
    pyproject_contents = pyproject.read_text()
    pyproject.write_text(
        pyproject_contents.replace("__package_name__", module_name).replace(
            "__module_name__", module_name
        )
    )

    # move module folder
    package_dir = destination_dir / module_name
    shutil.move(destination_dir / "package_template", package_dir)

    # replace readme
    readme = destination_dir / "README.md"
    readme_contents = readme.read_text()
    readme.write_text(
        readme_contents.replace("__package_name__", package_name).replace(
            "__app_route_code__", app_route_code
        )
    )

    # poetry install
    if with_poetry:
        subprocess.run(["poetry", "install"], cwd=destination_dir)


@package_cli.command()
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
    Starts a demo app for this template.
    """
    # load pyproject.toml
    project_dir = get_package_root()
    pyproject = project_dir / "pyproject.toml"

    # get langserve export - throws KeyError if invalid
    get_langserve_export(pyproject)

    port_str = str(port) if port is not None else "8000"
    host_str = host if host is not None else "127.0.0.1"

    command = [
        "uvicorn",
        "--factory",
        "langchain_cli.dev_scripts:create_demo_server",
        "--reload",
        "--port",
        port_str,
        "--host",
        host_str,
    ]
    subprocess.run(command)
