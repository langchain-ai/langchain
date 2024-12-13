"""
Develop integration packages for LangChain.
"""

import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, cast

import typer
from typing_extensions import Annotated, TypedDict

from langchain_cli.utils.find_replace import replace_file, replace_glob

integration_cli = typer.Typer(no_args_is_help=True, add_completion=False)


class Replacements(TypedDict):
    __package_name__: str
    __module_name__: str
    __ModuleName__: str
    __MODULE_NAME__: str
    __package_name_short__: str
    __package_name_short_snake__: str


def _process_name(name: str, *, community: bool = False) -> Replacements:
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
    replacements: Replacements = {
        "__package_name__": f"langchain-{preprocessed}",
        "__module_name__": "langchain_" + preprocessed.replace("-", "_"),
        "__ModuleName__": preprocessed.title().replace("-", ""),
        "__MODULE_NAME__": preprocessed.upper().replace("-", ""),
        "__package_name_short__": preprocessed,
        "__package_name_short_snake__": preprocessed.replace("-", "_"),
    }
    if community:
        replacements["__module_name__"] = preprocessed.replace("-", "_")
    return replacements


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
    src: Annotated[
        Optional[list[str]],
        typer.Option(
            help="The name of the single template file to copy."
            " e.g. `--src integration_template/chat_models.py "
            "--dst my_integration/chat_models.py`. Can be used multiple times.",
        ),
    ] = None,
    dst: Annotated[
        Optional[list[str]],
        typer.Option(
            help="The relative path to the integration package to place the new file in"
            ". e.g. `my-integration/my_integration.py`",
        ),
    ] = None,
):
    """
    Creates a new integration package.
    """

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

    project_template_dir = Path(__file__).parents[1] / "integration_template"
    destination_dir = Path.cwd() / replacements["__package_name__"]
    if not src and not dst:
        if destination_dir.exists():
            typer.echo(f"Folder {destination_dir} exists.")
            raise typer.Exit(code=1)

        # copy over template from ../integration_template
        shutil.copytree(project_template_dir, destination_dir, dirs_exist_ok=False)

        # folder movement
        package_dir = destination_dir / replacements["__module_name__"]
        shutil.move(destination_dir / "integration_template", package_dir)

        # replacements in files
        replace_glob(destination_dir, "**/*", cast(Dict[str, str], replacements))

        # poetry install
        subprocess.run(
            ["poetry", "install", "--with", "lint,test,typing,test_integration"],
            cwd=destination_dir,
        )
    else:
        # confirm src and dst are the same length
        if not src:
            typer.echo("Cannot provide --dst without --src.")
            raise typer.Exit(code=1)
        src_paths = [project_template_dir / p for p in src]
        if dst and len(src) != len(dst):
            typer.echo("Number of --src and --dst arguments must match.")
            raise typer.Exit(code=1)
        if not dst:
            # assume we're in a package dir, copy to equivalent path
            dst_paths = [destination_dir / p for p in src]
        else:
            dst_paths = [Path.cwd() / p for p in dst]
            dst_paths = [
                p / f"{replacements['__package_name_short_snake__']}.ipynb"
                if not p.suffix
                else p
                for p in dst_paths
            ]

        # confirm no duplicate dst_paths
        if len(dst_paths) != len(set(dst_paths)):
            typer.echo(
                "Duplicate destination paths provided or computed - please "
                "specify them explicitly with --dst."
            )
            raise typer.Exit(code=1)

        # confirm no files exist at dst_paths
        for dst_path in dst_paths:
            if dst_path.exists():
                typer.echo(f"File {dst_path} exists.")
                raise typer.Exit(code=1)

        for src_path, dst_path in zip(src_paths, dst_paths):
            shutil.copy(src_path, dst_path)
            replace_file(dst_path, cast(Dict[str, str], replacements))


TEMPLATE_MAP: dict[str, str] = {
    "ChatModel": "chat.ipynb",
    "DocumentLoader": "document_loaders.ipynb",
    "Tool": "tools.ipynb",
    "VectorStore": "vectorstores.ipynb",
    "Embeddings": "text_embedding.ipynb",
    "ByteStore": "kv_store.ipynb",
    "LLM": "llms.ipynb",
    "Provider": "provider.ipynb",
    "Toolkit": "toolkits.ipynb",
    "Retriever": "retrievers.ipynb",
}

_component_types_str = ", ".join(f"`{k}`" for k in TEMPLATE_MAP.keys())


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
                f"The type of component. Currently supported: {_component_types_str}."
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
    if component_type not in TEMPLATE_MAP:
        typer.echo(
            f"Unrecognized {component_type=}. Expected one of {_component_types_str}."
        )
        raise typer.Exit(code=1)

    new(
        name=name,
        name_class=name_class,
        src=[f"docs/{TEMPLATE_MAP[component_type]}"],
        dst=[destination_dir],
    )
