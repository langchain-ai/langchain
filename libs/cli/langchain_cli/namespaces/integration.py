"""
Develop integration packages for LangChain.
"""

import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated, TypedDict

from langchain_cli.utils.find_replace import replace_file, replace_glob

integration_cli = typer.Typer(no_args_is_help=True, add_completion=False)

Replacements = TypedDict(
    "Replacements",
    {
        "__package_name__": str,
        "__module_name__": str,
        "__ModuleName__": str,
        "__MODULE_NAME__": str,
        "__package_name_short__": str,
    },
)


def _process_name(name: str, *, community: bool = False):
    preprocessed = name.replace("_", "-").lower()

    if preprocessed.startswith("langchain-"):
        preprocessed = preprocessed[len("langchain-") :]

    if not re.match(r"^[a-z][a-z0-9-]*$", preprocessed):
        raise ValueError(
            "Name should only contain lowercase letters (a-z), numbers, and hyphens"
            ", and start with a letter."
        )
    if preprocessed.endswith("-"):
        raise ValueError("Name should not end with `-`.")
    if preprocessed.find("--") != -1:
        raise ValueError("Name should not contain consecutive hyphens.")
    replacements = {
        "__package_name__": f"langchain-{preprocessed}",
        "__module_name__": "langchain_" + preprocessed.replace("-", "_"),
        "__ModuleName__": preprocessed.title().replace("-", ""),
        "__MODULE_NAME__": preprocessed.upper().replace("-", ""),
        "__package_name_short__": preprocessed,
        "__package_name_short_snake__": preprocessed.replace("-", "_"),
    }
    if community:
        replacements["__module_name__"] = preprocessed.replace("-", "_")
    return Replacements(replacements)


@integration_cli.command()
def new(
    name: Annotated[
        str,
        typer.Option(
            help="The name of the integration to create (e.g. `my-integration`)",
            prompt="The name of the integration to create (e.g. `my-integration`)",
        ),
    ],
    name_class: Annotated[
        Optional[str],
        typer.Option(
            help="The name of the integration in PascalCase. e.g. `MyIntegration`."
            " This is used to name classes like `MyIntegrationVectorStore`"
        ),
    ] = None,
):
    """
    Creates a new integration package.

    Should be run from libs/partners
    """
    # confirm that we are in the right directory
    if not Path.cwd().name == "partners" or not Path.cwd().parent.name == "libs":
        typer.echo(
            "This command should be run from the `libs/partners` directory in the "
            "langchain-ai/langchain monorepo. Continuing is NOT recommended."
        )
        typer.confirm("Are you sure you want to continue?", abort=True)

    try:
        replacements = _process_name(name)
    except ValueError as e:
        typer.echo(e)
        raise typer.Exit(code=1)

    if name_class:
        if not re.match(r"^[A-Z][a-zA-Z0-9]*$", name_class):
            typer.echo(
                "Name should only contain letters (a-z, A-Z), numbers, and underscores"
                ", and start with a capital letter."
            )
            raise typer.Exit(code=1)
        replacements["__ModuleName__"] = name_class
    else:
        replacements["__ModuleName__"] = typer.prompt(
            "Name of integration in PascalCase", default=replacements["__ModuleName__"]
        )

    destination_dir = Path.cwd() / replacements["__package_name_short__"]
    if destination_dir.exists():
        typer.echo(f"Folder {destination_dir} exists.")
        raise typer.Exit(code=1)

    # copy over template from ../integration_template
    project_template_dir = Path(__file__).parents[1] / "integration_template"
    shutil.copytree(project_template_dir, destination_dir, dirs_exist_ok=False)

    # folder movement
    package_dir = destination_dir / replacements["__module_name__"]
    shutil.move(destination_dir / "integration_template", package_dir)

    # replacements in files
    replace_glob(destination_dir, "**/*", replacements)

    # poetry install
    subprocess.run(
        ["poetry", "install", "--with", "lint,test,typing,test_integration"],
        cwd=destination_dir,
    )


@integration_cli.command()
def create_doc(
    name: Annotated[
        str,
        typer.Option(
            help=(
                "The kebab-case name of the integration (e.g. `openai`, "
                "`google-vertexai`). Do not include a 'langchain-' prefix."
            ),
            prompt=(
                "The kebab-case name of the integration (e.g. `openai`, "
                "`google-vertexai`). Do not include a 'langchain-' prefix."
            ),
        ),
    ],
    name_class: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "The PascalCase name of the integration (e.g. `OpenAI`, "
                "`VertexAI`). Do not include a 'Chat', 'VectorStore', etc. "
                "prefix/suffix."
            ),
        ),
    ] = None,
    component_type: Annotated[
        str,
        typer.Option(
            help=(
                "The type of component. Currently only 'ChatModel', "
                "'DocumentLoader', 'VectorStore' supported."
            ),
        ),
    ] = "ChatModel",
    destination_dir: Annotated[
        str,
        typer.Option(
            help="The relative path to the docs directory to place the new file in.",
            prompt="The relative path to the docs directory to place the new file in.",
        ),
    ] = "docs/docs/integrations/chat/",
):
    """
    Creates a new integration doc.
    """
    try:
        replacements = _process_name(name, community=component_type=="Tool")
    except ValueError as e:
        typer.echo(e)
        raise typer.Exit(code=1)

    if name_class:
        if not re.match(r"^[A-Z][a-zA-Z0-9]*$", name_class):
            typer.echo(
                "Name should only contain letters (a-z, A-Z), numbers, and underscores"
                ", and start with a capital letter."
            )
            raise typer.Exit(code=1)
        replacements["__ModuleName__"] = name_class
    else:
        replacements["__ModuleName__"] = typer.prompt(
            (
                "The PascalCase name of the integration (e.g. `OpenAI`, `VertexAI`). "
                "Do not include a 'Chat', 'VectorStore', etc. prefix/suffix."
            ),
            default=replacements["__ModuleName__"],
        )
    destination_path = (
        Path.cwd()
        / destination_dir
        / (replacements["__package_name_short_snake__"] + ".ipynb")
    )

    # copy over template from ../integration_template
    template_dir = Path(__file__).parents[1] / "integration_template" / "docs"
    if component_type == "ChatModel":
        docs_template = template_dir / "chat.ipynb"
    elif component_type == "DocumentLoader":
        docs_template = template_dir / "document_loaders.ipynb"
    elif component_type == "Tool":
        docs_template = template_dir / "tools.ipynb"
    elif component_type == "VectorStore":
        docs_template = template_dir / "vectorstores.ipynb"
    else:
        raise ValueError(
            f"Unrecognized {component_type=}. Expected one of 'ChatModel', "
            f"'DocumentLoader', 'Tool'."
        )
    shutil.copy(docs_template, destination_path)

    # replacements in file
    replace_file(destination_path, replacements)
