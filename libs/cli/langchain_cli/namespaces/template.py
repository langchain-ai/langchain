"""Develop installable templates."""

import re
import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from langchain_cli.utils.github import list_packages
from langchain_cli.utils.packages import get_langserve_export, get_package_root

package_cli = typer.Typer(no_args_is_help=True, add_completion=False)


@package_cli.command()
def new(
    name: Annotated[str, typer.Argument(help="The name of the folder to create")],
    with_poetry: Annotated[  # noqa: FBT002
        bool,
        typer.Option("--with-poetry/--no-poetry", help="Don't run poetry install"),
    ] = False,
) -> None:
    """Create a new template package."""
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
        pyproject_contents.replace("__package_name__", package_name).replace(
            "__module_name__",
            module_name,
        ),
    )

    # move module folder
    package_dir = destination_dir / module_name
    shutil.move(destination_dir / "package_template", package_dir)

    # update init
    init = package_dir / "__init__.py"
    init_contents = init.read_text()
    init.write_text(init_contents.replace("__module_name__", module_name))

    # replace readme
    readme = destination_dir / "README.md"
    readme_contents = readme.read_text()
    readme.write_text(
        readme_contents.replace("__package_name__", package_name).replace(
            "__app_route_code__",
            app_route_code,
        ),
    )

    # poetry install
    if with_poetry:
        subprocess.run(["poetry", "install"], cwd=destination_dir, check=True)  # noqa: S607


@package_cli.command()
def serve(
    *,
    port: Annotated[
        int | None,
        typer.Option(help="The port to run the server on"),
    ] = None,
    host: Annotated[
        str | None,
        typer.Option(help="The host to run the server on"),
    ] = None,
    configurable: Annotated[
        bool | None,
        typer.Option(
            "--configurable/--no-configurable",
            help="Whether to include a configurable route",
        ),
    ] = None,  # defaults to `not chat_playground`
    chat_playground: Annotated[
        bool,
        typer.Option(
            "--chat-playground/--no-chat-playground",
            help="Whether to include a chat playground route",
        ),
    ] = False,
) -> None:
    """Start a demo app for this template."""
    # load pyproject.toml
    project_dir = get_package_root()
    pyproject = project_dir / "pyproject.toml"

    # get langserve export - throws KeyError if invalid
    get_langserve_export(pyproject)

    host_str = host if host is not None else "127.0.0.1"

    script = (
        "langchain_cli.dev_scripts:create_demo_server_chat"
        if chat_playground
        else (
            "langchain_cli.dev_scripts:create_demo_server_configurable"
            if configurable
            else "langchain_cli.dev_scripts:create_demo_server"
        )
    )

    uvicorn.run(
        script,
        factory=True,
        reload=True,
        port=port if port is not None else 8000,
        host=host_str,
    )


@package_cli.command()
def list(contains: Annotated[str | None, typer.Argument()] = None) -> None:  # noqa: A001
    """List all or search for available templates."""
    packages = list_packages(contains=contains)
    for package in packages:
        typer.echo(package)
