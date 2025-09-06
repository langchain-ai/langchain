"""Migrate LangChain to the most recent version."""

import shutil
from pathlib import Path

import rich
import typer
from gritql import run  # type: ignore[import-untyped]
from typer import Option


def get_gritdir_path() -> Path:
    """Get the path to the grit directory.

    Uses a cache directory to avoid Git management conflicts when running
    in pyenv virtual environments where the .grit directory might be ignored
    by parent .gitignore files.
    """
    # Use user cache directory to avoid Git management conflicts
    cache_dir = Path.home() / ".cache" / "langchain-cli"
    cache_dir.mkdir(parents=True, exist_ok=True)

    grit_dir = cache_dir / ".grit"

    # If cached .grit directory doesn't exist, copy from package
    if not grit_dir.exists():
        source_grit_dir = Path(__file__).parent / ".grit"
        if source_grit_dir.exists():
            shutil.copytree(source_grit_dir, grit_dir)
        else:
            # Fallback to original behavior if source doesn't exist
            return source_grit_dir

    return grit_dir


def migrate(
    ctx: typer.Context,
    # Using diff instead of dry-run for backwards compatibility with the old CLI
    diff: bool = Option(  # noqa: FBT001
        False,  # noqa: FBT003
        "--diff",
        help="Show the changes that would be made without applying them.",
    ),
    interactive: bool = Option(  # noqa: FBT001
        False,  # noqa: FBT003
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
        "should add them to your dependencies! \n\n"
        '⚠️ This script is a "best-effort", and is likely to make some '
        "mistakes.\n\n"
        "🛡️ Backup your code prior to running the migration script -- it will "
        "modify your files!\n\n",
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
        grit_dir=str(get_gritdir_path()),
    )

    raise typer.Exit(code=final_code)
