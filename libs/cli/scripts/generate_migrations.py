# type: ignore
"""Script to generate migrations for the migration script."""

import json
import os
import pkgutil

import click

from langchain_cli.namespaces.migrate.generate.generic import (
    generate_simplified_migrations,
)
from langchain_cli.namespaces.migrate.generate.grit import (
    dump_migrations_as_grit,
)
from langchain_cli.namespaces.migrate.generate.partner import (
    get_migrations_for_partner_package,
)


@click.group()
def cli():
    """Migration scripts management."""
    pass


@cli.command()
@click.option(
    "--pkg1",
    default="langchain",
)
@click.option(
    "--pkg2",
    default="langchain_community",
)
@click.option(
    "--output",
    default=None,
    help="Output file for the migration script.",
)
@click.option(
    "--filter-by-all/--no-filter-by-all",
    default=True,
    help="Output file for the migration script.",
)
@click.option(
    "--format",
    type=click.Choice(["json", "grit"], case_sensitive=False),
    default="json",
    help="The output format for the migration script (json or grit).",
)
def generic(
    pkg1: str, pkg2: str, output: str, filter_by_all: bool, format: str
) -> None:
    """Generate a migration script."""
    click.echo("Migration script generated.")
    migrations = generate_simplified_migrations(pkg1, pkg2, filter_by_all=filter_by_all)

    if output is not None:
        name = output.removesuffix(".json").removesuffix(".grit")
    else:
        name = f"{pkg1}_to_{pkg2}"

    if output is None:
        output = f"{name}.json" if format == "json" else f"{name}.grit"

    if format == "json":
        dumped = json.dumps(migrations, indent=2, sort_keys=True)
    else:
        dumped = dump_migrations_as_grit(name, migrations)

    with open(output, "w") as f:
        f.write(dumped)


def handle_partner(pkg: str, output: str = None):
    migrations = get_migrations_for_partner_package(pkg)
    # Run with python 3.9+
    name = pkg.removeprefix("langchain_")
    data = dump_migrations_as_grit(name, migrations)
    output_name = f"{name}.grit" if output is None else output
    if migrations:
        with open(output_name, "w") as f:
            f.write(data)
        click.secho(f"LangChain migration script saved to {output_name}")
    else:
        click.secho(f"No migrations found for {pkg}", fg="yellow")


@cli.command()
@click.argument("pkg")
@click.option("--output", default=None, help="Output file for the migration script.")
def partner(pkg: str, output: str) -> None:
    """Generate migration scripts specifically for LangChain modules."""
    click.echo("Migration script for LangChain generated.")
    handle_partner(pkg, output)


@cli.command()
@click.argument("json_file")
def json_to_grit(json_file: str) -> None:
    """Generate a Grit migration from an old JSON migration file."""
    with open(json_file, "r") as f:
        migrations = json.load(f)
    name = os.path.basename(json_file).removesuffix(".json").removesuffix(".grit")
    data = dump_migrations_as_grit(name, migrations)
    output_name = f"{name}.grit"
    with open(output_name, "w") as f:
        f.write(data)
    click.secho(f"GritQL migration script saved to {output_name}")


@cli.command()
def all_installed_partner_pkgs() -> None:
    """Generate migration scripts for all LangChain modules."""
    # Will generate migrations for all partner packages.
    # Define as "langchain_<partner_name>".
    # First let's determine which packages are installed in the environment
    # and then generate migrations for them.
    langchain_pkgs = [
        name
        for _, name, _ in pkgutil.iter_modules()
        if name.startswith("langchain_")
        and name not in {"langchain_core", "langchain_cli", "langchain_community"}
    ]
    for pkg in langchain_pkgs:
        handle_partner(pkg)


if __name__ == "__main__":
    cli()
