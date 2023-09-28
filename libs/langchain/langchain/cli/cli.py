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
def callback() -> None:
    pass


AUTHOR_NAME_OPTION = typer.Option(default_factory=get_git_user_name, prompt=True)
AUTHOR_EMAIL_OPTION = typer.Option(default_factory=get_git_user_email, prompt=True)
USE_POETRY_OPTION = typer.Option(default_factory=is_poetry_installed)


@app.command()
def create_repo(
    project_directory: str,
    author_name: Annotated[str, AUTHOR_NAME_OPTION],
    author_email: Annotated[str, AUTHOR_EMAIL_OPTION],
    use_poetry: Annotated[bool, USE_POETRY_OPTION],
) -> None:
    return create(project_directory, author_name, author_email, use_poetry)
