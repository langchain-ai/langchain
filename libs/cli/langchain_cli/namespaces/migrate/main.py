"""Migrate LangChain to the most recent version."""

from pathlib import Path

import rich
import typer
from gritql import run  # type: ignore
from typer import Option


def get_gritdir_path() -> Path:
    """Get the path to the grit directory."""
    script_dir = Path(__file__).parent
    return script_dir / ".grit"


def migrate(
    ctx: typer.Context,
    # Using diff instead of dry-run for backwards compatibility with the old CLI
    diff: bool = Option(
        False,
        "--diff",
        help="Show the changes that would be made without applying them.",
    ),
    interactive: bool = Option(
        False,
        "--interactive",
        help="Prompt for confirmation before making each change",
    ),
) -> None:
    """Migrate langchain to the most recent version.

    Any undocumented arguments will be passed to the Grit CLI.
    """
    rich.print(
        "✈️ This script will help you migrate to a LangChain 0.3. "
        "This migration script will attempt to replace old imports in the code "
        "with new ones. "
        "If you need to migrate to LangChain 0.2, please downgrade to version 0.0.29 "
        "of the langchain-cli.\n\n"
        "🔄 You will need to run the migration script TWICE to migrate (e.g., "
        "to update llms import from langchain, the script will first move them to "
        "corresponding imports from the community package, and on the second "
        "run will migrate from the community package to the partner package "
        "when possible). \n\n"
        "🔍 You can pre-view the changes by running with the --diff flag. \n\n"
        "🚫 You can disable specific import changes by using the --disable "
        "flag. \n\n"
        "📄 Update your pyproject.toml or requirements.txt file to "
        "reflect any imports from new packages. For example, if you see new "
        "imports from langchain_openai, langchain_anthropic or "
        "langchain_text_splitters you "
        "should them to your dependencies! \n\n"
        '⚠️ This script is a "best-effort", and is likely to make some '
        "mistakes.\n\n"
        "🛡️ Backup your code prior to running the migration script -- it will "
        "modify your files!\n\n"
    )
    rich.print("-" * 10)
    rich.print()

    args = list(ctx.args)
    if interactive:
        args.append("--interactive")
    if diff:
        args.append("--dry-run")

    final_code = run.apply_pattern(
        "langchain_all_migrations()",
        args,
        grit_dir=get_gritdir_path(),
    )

    raise typer.Exit(code=final_code)
