import os
import string
import subprocess
from pathlib import Path
from typing import List, Sequence, Union

import typer

import langchain
from langchain.cli.create_repo.check_name import is_name_taken, lint_name


class UnderscoreTemplate(string.Template):
    delimiter = "____"


def get_git_user_name() -> Union[str, None]:
    try:
        return (
            subprocess.run(["git", "config", "--get", "user.name"], capture_output=True)
            .stdout.decode()
            .strip()
        )
    except FileNotFoundError:
        return None


def get_git_user_email() -> Union[str, None]:
    try:
        return (
            subprocess.run(
                ["git", "config", "--get", "user.email"], capture_output=True
            )
            .stdout.decode()
            .strip()
        )
    except FileNotFoundError:
        return None


def create(
    project_directory: str,
    author_name: str,
    author_email: str,
    use_poetry: bool,
) -> None:
    """Create a new LangChain project."""

    project_directory_path = Path(project_directory)
    project_name = project_directory_path.name
    project_name_identifier = project_name.replace("-", "_")

    _validate_name(project_name, project_name_identifier)

    typer.echo(
        f"\n{typer.style('1.', bold=True, fg=typer.colors.GREEN)} Creating new"
        f" LangChain project {typer.style(project_name, fg=typer.colors.BRIGHT_CYAN)}"
        f" in"
        f" {typer.style(project_directory_path.resolve(), fg=typer.colors.BRIGHT_CYAN)}"
    )

    _create_project_dir(
        project_directory_path,
        use_poetry,
        project_name,
        project_name_identifier,
        author_name,
        author_email,
    )

    if use_poetry:
        _poetry_install(project_directory_path)
    else:
        _pip_install(project_directory_path)

    _init_git(project_directory_path)

    typer.echo(
        f"\n{typer.style('Done!', bold=True, fg=typer.colors.GREEN)}"
        f" Your new LangChain project"
        f" {typer.style(project_name, fg=typer.colors.BRIGHT_CYAN)}"
        f" has been created in"
        f" {typer.style(project_directory_path.resolve(), fg=typer.colors.BRIGHT_CYAN)}"
        f"."
    )
    cd_dir = typer.style(
        f"cd {project_directory_path.resolve()}", fg=typer.colors.BRIGHT_CYAN
    )
    typer.echo(
        f"\nChange into the project directory with {cd_dir}."
        f" The following commands are available:"
    )
    subprocess.run(["make"], cwd=project_directory_path)

    if not use_poetry:
        pip_install = typer.style(
            'pip install -e ".[dev]"', fg=typer.colors.BRIGHT_CYAN
        )
        typer.echo(
            f"\nTo install all dependencies activate your environment run:"
            f"\n{typer.style('source .venv/bin/activate', fg=typer.colors.BRIGHT_CYAN)}"
            f"\n{pip_install}."
        )


def is_poetry_installed() -> bool:
    return subprocess.run(["poetry", "--version"], capture_output=True).returncode == 0


def _validate_name(project_name: str, project_name_identifier: str) -> None:
    project_name_diagnostics = lint_name(project_name_identifier)
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
        typer.echo(
            "Please choose another name and try again.",
            err=True,
        )
        raise typer.Exit(code=1)
    if is_name_taken(project_name):
        typer.echo(
            f"{typer.style('Error:', fg=typer.colors.RED)}"
            f" The project name"
            f" {typer.style(project_name, fg=typer.colors.BRIGHT_CYAN)}"
            f" is already taken.",
            err=True,
        )
        typer.echo(
            "Please choose another name and try again.",
            err=True,
        )
        raise typer.Exit(code=1)


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
                    f" that  would be overwritten by the template.",
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
    typer.echo(
        f"\n{typer.style('2.', bold=True, fg=typer.colors.GREEN)}"
        f" Creating virtual environment..."
    )
    subprocess.run(["pwd"], cwd=project_directory_path)
    subprocess.run(["python", "-m", "venv", ".venv"], cwd=project_directory_path)


def _init_git(project_directory_path: Path) -> None:
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
