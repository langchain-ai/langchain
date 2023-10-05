""""""
import os
import pathlib
import string
import subprocess
from pathlib import Path
from typing import List, Sequence

import typer

import langchain
from langchain.cli.create_repo.pypi_name import is_name_taken, lint_name


class UnderscoreTemplate(string.Template):
    delimiter = "____"


def _create_project_dir(
    project_directory_path: Path,
    use_poetry: bool,
    project_name: str,
    project_name_identifier: str,
    author_name: str,
    author_email: str,
) -> None:
    project_directory_path.mkdir(parents=True, exist_ok=True)
    template_directories = _get_template_directories(use_poetry)
    _check_conflicting_files(template_directories, project_directory_path)
    _copy_template_files(
        template_directories,
        project_directory_path,
        project_name,
        project_name_identifier,
        author_name,
        author_email,
    )


def _get_template_directories(use_poetry: bool) -> List[Path]:
    """Get the directories containing the templates.

    Args:
        use_poetry: If true, will set up the project with Poetry.

    """
    template_parent_path = Path(__file__).parent / "templates"
    template_directories = [template_parent_path / "repo"]
    if use_poetry:
        template_directories.append(template_parent_path / "poetry")
    else:
        template_directories.append(template_parent_path / "pip")
    return template_directories


def _check_conflicting_files(
    template_directories: Sequence[Path], project_directory_path: Path
) -> None:
    """Validate project directory doesn't contain conflicting files."""

    for template_directory_path in template_directories:
        for template_file_path in template_directory_path.glob("**/*"):
            relative_template_file_path = template_file_path.relative_to(
                template_directory_path
            )
            project_file_path = project_directory_path / relative_template_file_path
            if project_file_path.exists():
                typer.echo(
                    f"{typer.style('Error:', fg=typer.colors.RED)}"
                    f" The project directory already contains a file"
                    f" {typer.style(project_file_path, fg=typer.colors.BRIGHT_CYAN)}"
                    f" that would be overwritten by the template.",
                    err=True,
                )
                typer.echo(
                    "Please remove this file and try again.",
                    err=True,
                )
                raise typer.Exit(code=1)


def _copy_template_files(
    template_directories: Sequence[Path],
    project_directory_path: Path,
    project_name: str,
    project_name_identifier: str,
    author_name: str,
    author_email: str,
) -> None:
    """Copy template files to project directory and substitute variables.

    Args:
        template_directories: The directories containing the templates.
        project_directory_path: The destination directory.
        project_name: The name of the project.
        project_name_identifier: The identifier of the project name.
        author_name: The name of the author.
        author_email: The email of the author.
    """
    for template_directory_path in template_directories:
        for template_file_path in template_directory_path.glob("**/*"):
            relative_template_file_path = UnderscoreTemplate(
                str(template_file_path.relative_to(template_directory_path))
            ).substitute(project_name_identifier=project_name_identifier)
            project_file_path = project_directory_path / relative_template_file_path
            if template_file_path.is_dir():
                project_file_path.mkdir(parents=True, exist_ok=True)
            else:
                project_file_path.write_text(
                    UnderscoreTemplate(template_file_path.read_text()).substitute(
                        project_name=project_name,
                        project_name_identifier=project_name_identifier,
                        author_name=author_name,
                        author_email=author_email,
                        langchain_version=langchain.__version__,
                    )
                )


def _poetry_install(project_directory_path: Path) -> None:
    """Install dependencies with Poetry."""
    typer.echo(
        f"\n{typer.style('2.', bold=True, fg=typer.colors.GREEN)}"
        f" Installing dependencies with Poetry..."
    )
    subprocess.run(["pwd"], cwd=project_directory_path)
    subprocess.run(
        ["poetry", "install"],
        cwd=project_directory_path,
        env={**os.environ.copy(), "VIRTUAL_ENV": ""},
    )


def _pip_install(project_directory_path: Path) -> None:
    """Create virtual environment and install dependencies."""
    typer.echo(
        f"\n{typer.style('2.', bold=True, fg=typer.colors.GREEN)}"
        f" Creating virtual environment..."
    )
    subprocess.run(["pwd"], cwd=project_directory_path)
    subprocess.run(["python", "-m", "venv", ".venv"], cwd=project_directory_path)
    # TODO install dependencies


def _init_git(project_directory_path: Path) -> None:
    """Initialize git repository."""
    typer.echo(
        f"\n{typer.style('3.', bold=True, fg=typer.colors.GREEN)} Initializing git..."
    )
    subprocess.run(["git", "init"], cwd=project_directory_path)

    # 7. Create initial commit
    subprocess.run(["git", "add", "."], cwd=project_directory_path)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=project_directory_path,
    )


def _select_project_name(suggested_project_name: str) -> str:
    """Help the user select a valid project name."""
    while True:
        project_name = typer.prompt(
            "Please choose a project name: ", default=suggested_project_name
        )

        project_name_diagnostics = lint_name(project_name)
        if project_name_diagnostics:
            typer.echo(
                f"{typer.style('Error:', fg=typer.colors.RED)}"
                f" The project name"
                f" {typer.style(project_name, fg=typer.colors.BRIGHT_CYAN)}"
                f" is not valid:",
                err=True,
            )

            for diagnostic in project_name_diagnostics:
                typer.echo(f"  - {diagnostic}")

            if typer.confirm(
                "Would you like to choose another name? "
                "Choose NO to proceed with existing name.",
                default=True,
            ):
                continue

        if is_name_taken(project_name):
            typer.echo(
                f"{typer.style('Error:', fg=typer.colors.RED)}"
                f" The project name"
                f" {typer.style(project_name, fg=typer.colors.BRIGHT_CYAN)}"
                f" is already taken on pypi",
                err=True,
            )

            if typer.confirm(
                "Would you like to choose another name? "
                "Choose NO to proceed with existing name.",
                default=True,
            ):
                continue

        # If we got here then the project name is valid and not taken
        return project_name


# PUBLIC API


def create(
    project_directory: pathlib.Path,
    author_name: str,
    author_email: str,
    use_poetry: bool,
) -> None:
    """Create a new LangChain project.

    Args:
        project_directory (str): The directory to create the project in.
        author_name (str): The name of the author.
        author_email (str): The email of the author.
        use_poetry (bool): Whether to use Poetry to manage the project.
    """

    project_directory_path = Path(project_directory)
    project_name_suggestion = project_directory_path.name.replace("-", "_")
    project_name = _select_project_name(project_name_suggestion)
    project_name_identifier = project_name

    resolved_path = project_directory_path.resolve()

    if not typer.confirm(
        f"\n{typer.style('>', bold=True, fg=typer.colors.GREEN)} "
        f"Creating new LangChain project "
        f"{typer.style(project_name, fg=typer.colors.BRIGHT_CYAN)}"
        f" in"
        f" {typer.style(resolved_path, fg=typer.colors.BRIGHT_CYAN)}",
        default=True,
    ):
        typer.echo("OK! Canceling project creation.")
        raise typer.Exit(code=0)

    _create_project_dir(
        project_directory_path,
        use_poetry,
        project_name,
        project_name_identifier,
        author_name,
        author_email,
    )

    # TODO(Team): Add installation
    # if use_poetry:
    #     _poetry_install(project_directory_path)
    # else:
    #     _pip_install(project_directory_path)

    _init_git(project_directory_path)

    typer.echo(
        f"\n{typer.style('Done!', bold=True, fg=typer.colors.GREEN)}"
        f" Your new LangChain project"
        f" {typer.style(project_name, fg=typer.colors.BRIGHT_CYAN)}"
        f" has been created in"
        f" {typer.style(project_directory_path.resolve(), fg=typer.colors.BRIGHT_CYAN)}"
        f"."
    )
    # TODO(Team): Add surfacing information from make file and installation
    # cd_dir = typer.style(
    #     f"cd {project_directory_path.resolve()}", fg=typer.colors.BRIGHT_CYAN
    # )
    # typer.echo(
    #     f"\nChange into the project directory with {cd_dir}."
    #     f" The following commands are available:"
    # )
    # subprocess.run(["make"], cwd=project_directory_path)

    # if not use_poetry:
    #     pip_install = typer.style(
    #         'pip install -e ".[dev]"', fg=typer.colors.BRIGHT_CYAN
    #     )
    #     typer.echo(
    #         f"\nTo install all dependencies activate your environment run:"
    #     f"\n{typer.style('source .venv/bin/activate', fg=typer.colors.BRIGHT_CYAN)}"
    #         f"\n{pip_install}."
    #     )


def is_poetry_installed() -> bool:
    """Check if Poetry is installed."""
    return subprocess.run(["poetry", "--version"], capture_output=True).returncode == 0
