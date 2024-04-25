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
from typing import Any, Dict, Iterable, List, Tuple, Type, TypeVar, Union

import libcst as cst
import rich
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
):
    """Migrate langchain to the most recent version."""
    if not diff:
        rich.print("[bold red]Alert![/ bold red] langchain-cli migrate", end=": ")
        if not typer.confirm(
            "The migration process will modify your files. "
            "The migration is a `best-effort` process and is not expected to "
            "be perfect. "
            "Do you want to continue?"
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

    codemods = gather_codemods(disabled=disable)

    log_fp = log_file.open("a+", encoding="utf8")
    partial_run_codemods = functools.partial(
        run_codemods, codemods, metadata_manager, scratch, package, diff
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
