"""Migrate LangChain to the most recent version."""

# Adapted from bump-pydantic
# https://github.com/pydantic/bump-pydantic
import difflib
import functools
import multiprocessing
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union

import libcst as cst
import typer
from libcst.codemod import CodemodContext, ContextAwareTransformer
from libcst.helpers import calculate_module_and_package
from libcst.metadata import FullRepoManager, FullyQualifiedNameProvider, ScopeProvider
from rich.console import Console
from rich.progress import Progress
from typer import Argument, Exit, Option, Typer
from typing_extensions import ParamSpec

from langchain_cli.namespaces.migrate.codemods import Rule, gather_codemods
from langchain_cli.namespaces.migrate.glob_helpers import match_glob

app = Typer(invoke_without_command=True, add_completion=False)

P = ParamSpec("P")
T = TypeVar("T")

DEFAULT_IGNORES = [".venv/**"]


@app.callback()
def main(
    path: Path = Argument(..., exists=True, dir_okay=True, allow_dash=False),
    disable: List[Rule] = Option(default=[], help="Disable a rule."),
    diff: bool = Option(False, help="Show diff instead of applying changes."),
    ignore: List[str] = Option(
        default=DEFAULT_IGNORES, help="Ignore a path glob pattern."
    ),
    log_file: Path = Option("log.txt", help="Log errors to this file."),
    include_ipynb: bool = Option(
        False, help="Include Jupyter Notebook files in the migration."
    ),
):
    """Migrate langchain to the most recent version."""
    if not diff:
        if not typer.confirm(
            "✈️ This script will help you migrate to a recent version LangChain. "
            "This migration script will attempt to replace old imports in the code "
            "with new ones.\n\n"
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
            "❓ Do you want to continue?"
        ):
            raise Exit()
    console = Console(log_time=True)
    console.log("Start langchain-cli migrate")
    # NOTE: LIBCST_PARSER_TYPE=native is required according to https://github.com/Instagram/LibCST/issues/487.
    os.environ["LIBCST_PARSER_TYPE"] = "native"

    if os.path.isfile(path):
        package = path.parent
        all_files = [path]
    else:
        package = path
        all_files = sorted(package.glob("**/*.py"))
        if include_ipynb:
            all_files.extend(sorted(package.glob("**/*.ipynb")))

    filtered_files = [
        file
        for file in all_files
        if not any(match_glob(file, pattern) for pattern in ignore)
    ]
    files = [str(file.relative_to(".")) for file in filtered_files]

    if len(files) == 1:
        console.log("Found 1 file to process.")
    elif len(files) > 1:
        console.log(f"Found {len(files)} files to process.")
    else:
        console.log("No files to process.")
        raise Exit()

    providers = {FullyQualifiedNameProvider, ScopeProvider}
    metadata_manager = FullRepoManager(".", files, providers=providers)  # type: ignore[arg-type]
    metadata_manager.resolve_cache()

    scratch: dict[str, Any] = {}
    start_time = time.time()

    log_fp = log_file.open("a+", encoding="utf8")
    partial_run_codemods = functools.partial(
        get_and_run_codemods, disable, metadata_manager, scratch, package, diff
    )
    with Progress(*Progress.get_default_columns(), transient=True) as progress:
        task = progress.add_task(description="Executing codemods...", total=len(files))
        count_errors = 0
        difflines: List[List[str]] = []
        with multiprocessing.Pool() as pool:
            for error, _difflines in pool.imap_unordered(partial_run_codemods, files):
                progress.advance(task)

                if _difflines is not None:
                    difflines.append(_difflines)

                if error is not None:
                    count_errors += 1
                    log_fp.writelines(error)

    modified = [Path(f) for f in files if os.stat(f).st_mtime > start_time]

    if not diff:
        if modified:
            console.log(f"Refactored {len(modified)} files.")
        else:
            console.log("No files were modified.")

    for _difflines in difflines:
        color_diff(console, _difflines)

    if count_errors > 0:
        console.log(f"Found {count_errors} errors. Please check the {log_file} file.")
    else:
        console.log("Run successfully!")

    if difflines:
        raise Exit(1)


def get_and_run_codemods(
    disabled_rules: List[Rule],
    metadata_manager: FullRepoManager,
    scratch: Dict[str, Any],
    package: Path,
    diff: bool,
    filename: str,
) -> Tuple[Union[str, None], Union[List[str], None]]:
    """Run codemods from rules.

    Wrapper around run_codemods to be used with multiprocessing.Pool.
    """
    codemods = gather_codemods(disabled=disabled_rules)
    return run_codemods(codemods, metadata_manager, scratch, package, diff, filename)


def _rewrite_file(
    filename: str,
    codemods: List[Type[ContextAwareTransformer]],
    diff: bool,
    context: CodemodContext,
) -> Tuple[Union[str, None], Union[List[str], None]]:
    file_path = Path(filename)
    with file_path.open("r+", encoding="utf-8") as fp:
        code = fp.read()
        fp.seek(0)

        input_tree = cst.parse_module(code)

        for codemod in codemods:
            transformer = codemod(context=context)
            output_tree = transformer.transform_module(input_tree)
            input_tree = output_tree

        output_code = input_tree.code
        if code != output_code:
            if diff:
                lines = difflib.unified_diff(
                    code.splitlines(keepends=True),
                    output_code.splitlines(keepends=True),
                    fromfile=filename,
                    tofile=filename,
                )
                return None, list(lines)
            else:
                fp.write(output_code)
                fp.truncate()
    return None, None


def _rewrite_notebook(
    filename: str,
    codemods: List[Type[ContextAwareTransformer]],
    diff: bool,
    context: CodemodContext,
) -> Tuple[Optional[str], Optional[List[str]]]:
    """Try to rewrite a Jupyter Notebook file."""
    import nbformat

    file_path = Path(filename)
    if file_path.suffix != ".ipynb":
        raise ValueError("Only Jupyter Notebook files (.ipynb) are supported.")

    with file_path.open("r", encoding="utf-8") as fp:
        notebook = nbformat.read(fp, as_version=4)

    diffs = []

    for cell in notebook.cells:
        if cell.cell_type == "code":
            code = "".join(cell.source)

            # Skip code if any of the lines begin with a magic command or
            # a ! command.
            # We can try to handle later.
            if any(
                line.startswith("!") or line.startswith("%")
                for line in code.splitlines()
            ):
                continue

            input_tree = cst.parse_module(code)

            # TODO(Team): Quick hack, need to figure out
            # how to handle this correctly.
            # This prevents the code from trying to re-insert the imports
            # for every cell in the notebook.
            local_context = CodemodContext()

            for codemod in codemods:
                transformer = codemod(context=local_context)
                output_tree = transformer.transform_module(input_tree)
                input_tree = output_tree

            output_code = input_tree.code
            if code != output_code:
                cell.source = output_code.splitlines(keepends=True)
                if diff:
                    cell_diff = difflib.unified_diff(
                        code.splitlines(keepends=True),
                        output_code.splitlines(keepends=True),
                        fromfile=filename,
                        tofile=filename,
                    )
                    diffs.extend(list(cell_diff))

    if diff:
        return None, diffs

    with file_path.open("w", encoding="utf-8") as fp:
        nbformat.write(notebook, fp)

    return None, None


def run_codemods(
    codemods: List[Type[ContextAwareTransformer]],
    metadata_manager: FullRepoManager,
    scratch: Dict[str, Any],
    package: Path,
    diff: bool,
    filename: str,
) -> Tuple[Union[str, None], Union[List[str], None]]:
    try:
        module_and_package = calculate_module_and_package(str(package), filename)
        context = CodemodContext(
            metadata_manager=metadata_manager,
            filename=filename,
            full_module_name=module_and_package.name,
            full_package_name=module_and_package.package,
        )
        context.scratch.update(scratch)

        if filename.endswith(".ipynb"):
            return _rewrite_notebook(filename, codemods, diff, context)
        else:
            return _rewrite_file(filename, codemods, diff, context)
    except cst.ParserSyntaxError as exc:
        return (
            f"A syntax error happened on {filename}. This file cannot be "
            f"formatted.\n"
            f"{exc}"
        ), None
    except Exception:
        return f"An error happened on {filename}.\n{traceback.format_exc()}", None


def color_diff(console: Console, lines: Iterable[str]) -> None:
    for line in lines:
        line = line.rstrip("\n")
        if line.startswith("+"):
            console.print(line, style="green")
        elif line.startswith("-"):
            console.print(line, style="red")
        elif line.startswith("^"):
            console.print(line, style="blue")
        else:
            console.print(line, style="white")
