"""A CLI for creating a new project with LangChain."""
from pathlib import Path

from langchain.cli.create_repo.pypi_name import lint_name, is_name_taken

try:
    import typer
except ImportError:
    raise ImportError(
        "Typer must be installed to use the CLI. "
        "You can install it with `pip install typer`."
    )

from typing_extensions import Annotated

from langchain.cli.create_repo.base import create, is_poetry_installed
from langchain.cli.create_repo.user_info import get_git_user_email, get_git_user_name

app = typer.Typer(no_args_is_help=False, add_completion=False)


AUTHOR_NAME_OPTION = typer.Option(
    default_factory=get_git_user_name,
    prompt=True,
    help="If not specified, will be inferred from git config if possible. ",
)
AUTHOR_EMAIL_OPTION = typer.Option(
    default_factory=get_git_user_email,
    prompt=True,
    help="If not specified, will be inferred from git config if possible. ",
)
USE_POETRY_OPTION = typer.Option(
    default_factory=is_poetry_installed,
    prompt=True,
    help=(
        "Whether to use Poetry to manage the project. "
        "If not specified, Poetry will be used if poetry is installed."
    ),
)

PROJECT_NAME = typer.Option(
    None,
    prompt=True,
    help="The name of the project. If not specified, will be inferred from the directory name.",
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


@app.command()
def new(
    project_directory: Annotated[
        Path, typer.Argument(help="The directory to create the project in.")
    ],
    # author_name: Annotated[str, AUTHOR_NAME_OPTION],
    # author_email: Annotated[str, AUTHOR_EMAIL_OPTION],
    # use_poetry: Annotated[bool, USE_POETRY_OPTION],
) -> None:
    """Create a new project with LangChain."""
    project_directory_path = Path(project_directory)
    project_name = typer.prompt("Project Name", default=project_directory)
    project_name_suggestion = project_directory_path.name.replace("-", "_")
    project_name = _select_project_name(project_name_suggestion)
    project_name_identifier = project_name
    raise ValueError(project_name)
    create(project_directory, author_name, author_email, use_poetry)


if __name__ == "__main__":
    app()
