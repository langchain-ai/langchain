"""A CLI for creating a new project with LangChain."""
import typer
from typing_extensions import Annotated

from langchain.cli.create_repo import (
    create,
    get_git_user_email,
    get_git_user_name,
    is_poetry_installed,
)

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Create a new project with LangChain."""


AUTHOR_NAME_OPTION = typer.Option(default_factory=get_git_user_name, prompt=True)
AUTHOR_EMAIL_OPTION = typer.Option(default_factory=get_git_user_email, prompt=True)
USE_POETRY_OPTION = typer.Option(default_factory=is_poetry_installed)


@app.command()
def new(
    project_directory: str,
    author_name: Annotated[str, AUTHOR_NAME_OPTION],
    author_email: Annotated[str, AUTHOR_EMAIL_OPTION],
    use_poetry: Annotated[bool, USE_POETRY_OPTION],
) -> None:
    """Create a new project with LangChain.

    Args:
        project_directory (str): The directory to create the project in.
        author_name (str): The name of the author.
        author_email (str): The email of the author.
        use_poetry (bool): Whether to use Poetry to manage the project.
    """
    return create(project_directory, author_name, author_email, use_poetry)


if __name__ == "__main__":
    main()
