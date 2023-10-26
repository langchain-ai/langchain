"""
Manage LangServe application projects.
"""

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

import tomli
import typer
from langserve.packages import get_langserve_export, list_packages
from typing_extensions import Annotated

from langchain_cli.utils.events import create_events
from langchain_cli.utils.git import (
    copy_repo,
    parse_dependency_string,
    update_repo,
)
from langchain_cli.utils.packages import get_package_root

REPO_DIR = Path(typer.get_app_dir("langchain")) / "git_repos"

serve = typer.Typer(no_args_is_help=True, add_completion=False)


@serve.command()
def new(
    name: Annotated[str, typer.Argument(help="The name of the folder to create")],
    *,
    package: Annotated[
        Optional[List[str]],
        typer.Option(help="Packages to seed the project with"),
    ] = None,
    with_poetry: Annotated[
        bool,
        typer.Option("--with-poetry/--no-poetry", help="Run poetry install"),
    ] = False,
):
    """
    Create a new LangServe application.
    """
    # copy over template from ../project_template
    project_template_dir = Path(__file__).parents[1] / "project_template"
    destination_dir = Path.cwd() / name if name != "." else Path.cwd()
    shutil.copytree(project_template_dir, destination_dir, dirs_exist_ok=name == ".")

    # poetry install
    if with_poetry:
        subprocess.run(["poetry", "install"], cwd=destination_dir)

    # add packages if specified
    if package is not None and len(package) > 0:
        add(package, project_dir=destination_dir, with_poetry=with_poetry)


@serve.command()
def install():
    package_root = get_package_root() / "packages"
    for package_path in list_packages(package_root):
        try:
            pyproject_path = package_path / "pyproject.toml"
            langserve_export = get_langserve_export(pyproject_path)
            typer.echo(f"Installing {langserve_export['package_name']}...")
            subprocess.run(["poetry", "add", "--editable", package_path])
        except Exception as e:
            typer.echo(f"Skipping installing {package_path} due to error: {e}")


@serve.command()
def add(
    dependencies: Annotated[
        Optional[List[str]], typer.Argument(help="The dependency to add")
    ] = None,
    *,
    api_path: Annotated[List[str], typer.Option(help="API paths to add")] = [],
    project_dir: Annotated[
        Optional[Path], typer.Option(help="The project directory")
    ] = None,
    repo: Annotated[
        List[str], typer.Option(help="Shorthand for installing a GitHub Repo")
    ] = [],
    with_poetry: Annotated[
        bool,
        typer.Option("--with-poetry/--no-poetry", help="Run poetry install"),
    ] = False,
):
    """
    Adds the specified package to the current LangServe instance.

    e.g.:
    langchain serve add simple-pirate
    langchain serve add git+ssh://git@github.com/efriis/simple-pirate.git
    langchain serve add git+https://github.com/efriis/hub.git#devbranch#subdirectory=mypackage
    """
    project_root = get_package_root(project_dir)

    if dependencies is None:
        dependencies = []

    # cannot have both repo and dependencies
    if len(repo) != 0:
        if len(dependencies) != 0:
            raise typer.BadParameter(
                "Cannot specify both repo and dependencies. "
                "Please specify one or the other."
            )
        dependencies = [f"git+https://github.com/{r}" for r in repo]

    if len(api_path) != 0 and len(api_path) != len(dependencies):
        raise typer.BadParameter(
            "The number of API paths must match the number of dependencies."
        )

    # get installed packages from pyproject.toml
    root_pyproject_path = project_root / "pyproject.toml"
    with open(root_pyproject_path, "rb") as pyproject_file:
        pyproject = tomli.load(pyproject_file)
    installed_packages = (
        pyproject.get("tool", {}).get("poetry", {}).get("dependencies", {})
    )
    installed_names = set(installed_packages.keys())

    package_dir = project_root / "packages"

    create_events(
        [{"event": "serve add", "properties": {"package": d}} for d in dependencies]
    )

    for i, dependency in enumerate(dependencies):
        # update repo
        typer.echo(f"Adding {dependency}...")
        dep = parse_dependency_string(dependency)
        source_repo_path = update_repo(dep["git"], dep["ref"], REPO_DIR)
        source_path = (
            source_repo_path / dep["subdirectory"]
            if dep["subdirectory"]
            else source_repo_path
        )
        pyproject_path = source_path / "pyproject.toml"
        if not pyproject_path.exists():
            typer.echo(f"Could not find {pyproject_path}")
            continue
        langserve_export = get_langserve_export(pyproject_path)

        # detect name conflict
        if langserve_export["package_name"] in installed_names:
            typer.echo(
                f"Package with name {langserve_export['package_name']} already "
                "installed. Skipping...",
            )
            continue

        inner_api_path = (
            api_path[i] if len(api_path) != 0 else langserve_export["package_name"]
        )
        destination_path = package_dir / inner_api_path
        if destination_path.exists():
            typer.echo(
                f"Endpoint {langserve_export['package_name']} already exists. "
                "Skipping...",
            )
            continue
        copy_repo(source_path, destination_path)
        # poetry install
        if with_poetry:
            subprocess.run(
                ["poetry", "add", "--editable", destination_path], cwd=project_root
            )


@serve.command()
def remove(
    api_paths: Annotated[List[str], typer.Argument(help="The API paths to remove")],
    with_poetry: Annotated[
        bool,
        typer.Option("--with_poetry/--no-poetry", help="Don't run poetry remove"),
    ] = False,
):
    """
    Removes the specified package from the current LangServe instance.
    """
    for api_path in api_paths:
        package_dir = Path.cwd() / "packages" / api_path
        if not package_dir.exists():
            typer.echo(f"Endpoint {api_path} does not exist. Skipping...")
            continue
        pyproject = package_dir / "pyproject.toml"
        langserve_export = get_langserve_export(pyproject)
        typer.echo(f"Removing {langserve_export['package_name']}...")
        if with_poetry:
            subprocess.run(["poetry", "remove", langserve_export["package_name"]])
        shutil.rmtree(package_dir)


@serve.command()
def list():
    """
    Lists all packages in the current LangServe instance.
    """
    package_root = get_package_root() / "packages"
    for package_path in list_packages(package_root):
        relative = package_path.relative_to(package_root)
        pyproject_path = package_path / "pyproject.toml"
        langserve_export = get_langserve_export(pyproject_path)
        typer.echo(
            f"{relative}: ({langserve_export['module']}.{langserve_export['attr']})"
        )


@serve.command()
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
    Starts the LangServe instance.
    """
    cmd = ["poetry", "run", "poe", "start"]
    if port is not None:
        cmd += ["--port", str(port)]
    if host is not None:
        cmd += ["--host", host]
    subprocess.run(cmd)
