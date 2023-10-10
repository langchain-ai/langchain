"""A CLI for creating a new project with LangChain."""
from pathlib import Path
from typing import Optional

from typing_extensions import Annotated

# Keep this import here so that we can check if Typer is installed
try:
    import typer
except ImportError:
    raise ImportError(
        "Typer must be installed to use the CLI. "
        "You can install it with `pip install typer` or install LangChain "
        'with the [cli] extra like `pip install "langchain[cli]"`.'
    )

from langchain.cli.create_repo.base import create, is_poetry_installed
from langchain.cli.create_repo.pypi_name import is_name_taken, lint_name
from langchain.cli.create_repo.user_info import get_git_user_email, get_git_user_name

app = typer.Typer(no_args_is_help=False, add_completion=False)


def _select_project_name(suggested_project_name: str) -> str:
    """Help the user select a valid project name."""
    while True:
        project_name = typer.prompt("Project Name", default=suggested_project_name)

        project_name_diagnostics = lint_name(project_name)
        if project_name_diagnostics:
            typer.echo(
                f"{typer.style('Warning:', fg=typer.colors.MAGENTA)}"
                f" The project name"
                f" {typer.style(project_name, fg=typer.colors.BRIGHT_CYAN)}"
                f" is not valid.",
                err=True,
            )

            for diagnostic in project_name_diagnostics:
                typer.echo(f"  - {diagnostic}")

            if typer.confirm(
                "Select another name?",
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
                "Select another name?",
                default=True,
            ):
                continue

        # If we got here then the project name is valid and not taken
        return project_name


#
#
@app.command()
def new(
    project_directory: Annotated[
        Path, typer.Argument(help="The directory to create the project in.")
    ],
    author_name: Optional[str] = None,
    author_email: Optional[str] = None,
    use_poetry: Annotated[
        Optional[bool], typer.Option(help="Specify whether to use Poetry or not.")
    ] = None,
) -> None:
    """Create a new project with LangChain."""

    project_directory_path = Path(project_directory)
    project_name_suggestion = project_directory_path.name.replace("-", "_")
    project_name = _select_project_name(project_name_suggestion)

    if not author_name:
        author_name = typer.prompt("Author Name", default=get_git_user_name())

    if not author_email:
        author_email = typer.prompt("Author Email", default=get_git_user_email())

    if use_poetry is None:
        if is_poetry_installed():
            typer.echo("🎉 Found Poetry installed. Project can be set up using poetry.")
            use_poetry = typer.confirm("Use Poetry? (no to use pip)", default=True)
        else:
            typer.echo("ℹ️ Could not find Poetry installed.")
            use_pip = typer.confirm("Use Pip? (no to use poetry)", default=True)
            use_poetry = not use_pip

    if author_name is None:
        raise typer.BadParameter("Author name is required")

    if author_email is None:
        raise typer.BadParameter("Author email is required")

    create(project_directory, project_name, author_name, author_email, use_poetry)


if __name__ == "__main__":
    app()
