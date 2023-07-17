import os
import pathlib
import string
import subprocess
from typing import Union

import typer

import langchain
from langchain.cli.create.check_name import is_name_taken, lint_name


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


def main(
    project_directory: str,
    author_name: str,
    author_email: str,
):
    """
    Create a new LangChain project.
    """

    # 0. Create template variables
    project_directory_path = pathlib.Path(project_directory)
    project_name = project_directory_path.name
    project_name_identifier = project_name.replace("-", "_")
    langchain_version = langchain.__version__
    is_poetry_installed = (
        subprocess.run(["poetry", "--version"], capture_output=True).returncode == 0
    )

    if not is_poetry_installed:
        typer.echo(
            f"{typer.style('Warning:', fg=typer.colors.YELLOW)} Poetry is not installed. See here how to install it: https://python-poetry.org/docs/#installing-with-the-official-installer",
            err=True,
        )
        raise typer.Exit(code=1)

    # 1. Validate project name
    project_name_diagnostics = lint_name(project_name_identifier)
    if project_name_diagnostics:
        typer.echo(
            f"{typer.style('Error:', fg=typer.colors.RED)} The project name {typer.style(project_name, fg=typer.colors.BLUE)} is not valid:",
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
            f"{typer.style('Error:', fg=typer.colors.RED)} The project name {typer.style(project_name, fg=typer.colors.BLUE)} is already taken.",
            err=True,
        )
        typer.echo(
            "Please choose another name and try again.",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(
        f"\n{typer.style('1.', bold=True, fg=typer.colors.GREEN)} Creating new LangChain project {typer.style(project_name, fg=typer.colors.BLUE)} in {typer.style(pathlib.Path(project_directory).resolve(), fg=typer.colors.BLUE)}"
    )

    # 2. Create project directory
    project_directory_path.mkdir(parents=True, exist_ok=True)

    # 3. Select templates
    template_parent_path = pathlib.Path(__file__).parent.parent
    template_directories = [
        template_parent_path / "create_template",
        (template_parent_path / "create_template_poetry")
        if is_poetry_installed
        else (template_parent_path / "create_template_pip"),
    ]

    # 3. Validate project directory doesn't contain conflicting files
    for template_directory_path in template_directories:
        for template_file_path in template_directory_path.glob("**/*"):
            relative_template_file_path = template_file_path.relative_to(
                template_directory_path
            )
            project_file_path = project_directory_path / relative_template_file_path
            if project_file_path.exists():
                typer.echo(
                    f"{typer.style('Error:', fg=typer.colors.RED)} The project directory already contains a file {typer.style(project_file_path, fg=typer.colors.BLUE)} that would be overwritten by the template.",
                    err=True,
                )
                typer.echo(
                    "Please remove this file and try again.",
                    err=True,
                )
                raise typer.Exit(code=1)

    # 4. Copy template files
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
                        langchain_version=langchain_version,
                    )
                )

    # 5. Install dependencies
    if is_poetry_installed:
        typer.echo(
            f"\n{typer.style('2.', bold=True, fg=typer.colors.GREEN)} Installing dependencies with Poetry..."
        )
        subprocess.run(["pwd"], cwd=project_directory_path)
        subprocess.run(
            ["poetry", "install"],
            cwd=project_directory_path,
            env={**os.environ.copy(), "VIRTUAL_ENV": ""},
        )
    else:
        pass
        # TODO support pip

    # 6. Initialize git
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

    # 9. Done
    typer.echo(
        f"\n{typer.style('Done!', bold=True, fg=typer.colors.GREEN)} Your new LangChain project {typer.style(project_name, fg=typer.colors.BLUE)} has been created in {typer.style(pathlib.Path(project_directory).resolve(), fg=typer.colors.BLUE)}."
    )
    typer.echo(
        f"\nChange into the project directory with {typer.style(f'cd {project_name}', fg=typer.colors.BLUE)}. The following commands are available:"
    )
    subprocess.run(["make"], cwd=project_directory_path)
