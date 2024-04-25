"""Script to generate migrations for the migration script."""
import json

import click

from langchain_cli.namespaces.migrate.generate.langchain import (
    generate_migrations_from_langchain_to_community,
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
    "--output",
    default="langchain_migrations.json",
    help="Output file for the migration script.",
)
def langchain(output: str) -> None:
    """Generate a migration script."""
    click.echo("Migration script generated.")
    migrations = generate_migrations_from_langchain_to_community()
    with open(output, "w") as f:
        f.write(json.dumps(migrations))


@cli.command()
@click.argument("pkg")
@click.option("--output", default=None, help="Output file for the migration script.")
def partner(pkg: str, output: str) -> None:
    """Generate migration scripts specifically for LangChain modules."""
    click.echo("Migration script for LangChain generated.")
    migrations = get_migrations_for_partner_package(pkg)
    output_name = f"partner_{pkg}.json" if output is None else output
    with open(output_name, "w") as f:
        f.write(json.dumps(migrations, indent=2, sort_keys=True))
    click.secho(f"LangChain migration script saved to {output_name}")


if __name__ == "__main__":
    cli()
