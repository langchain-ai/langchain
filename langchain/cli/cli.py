import typer
from typing_extensions import Annotated

import langchain.cli.create.create

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def callback():
    pass


@app.command()
def create(
    project_directory: str,
    author_name: Annotated[
        str,
        typer.Option(
            default_factory=langchain.cli.create.create.get_git_user_name, prompt=True
        ),
    ],
    author_email: Annotated[
        str,
        typer.Option(
            default_factory=langchain.cli.create.create.get_git_user_email, prompt=True
        ),
    ],
):
    return langchain.cli.create.create.main(
        project_directory, author_name, author_email
    )
