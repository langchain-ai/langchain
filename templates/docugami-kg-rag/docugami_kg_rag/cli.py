from docugami import Docugami
import typer
from typing import Optional

from docugami_kg_rag.helpers.indexing import index_docset

app = typer.Typer()
docugami_client = Docugami()


@app.command()
def main(
    docset_id: Optional[str] = typer.Option(None, help="ID of the docset to ingest"),
    all_docsets: bool = typer.Option(False, help="Flag to ingest all docsets"),
):
    docsets_response = docugami_client.docsets.list()

    if not docsets_response or not docsets_response.docsets:
        raise typer.Exit("The workspace corresponding to the provided DOCUGAMI_API_KEY does not have any docsets.")

    docsets = docsets_response.docsets
    docset_ids = [d.id for d in docsets]
    if all_docsets:
        selected_docsets = docset_ids
    elif docset_id:
        if docset_id not in docset_ids:
            raise typer.Exit(f"Error: Docset with ID {docset_id} does not exist.")
        selected_docsets = [docset_id]
    else:
        typer.echo("Your workspace contains the following Docsets:\n")
        for idx, docset in enumerate(docsets, start=1):
            print(f"{idx}: {docset.name} (ID: {docset.id})")
        user_input = typer.prompt(
            "\nPlease enter the number(s) of the docset(s) to index (comma-separated) or 'all' in index all docsets"
        )

        if user_input.lower() == "all":
            selected_docsets = docset_ids
        else:
            selected_indices = [int(i.strip()) for i in user_input.split(",")]
            selected_docsets = [docsets[idx - 1] for idx in selected_indices if 0 < idx <= len(docsets)]

    for docset in selected_docsets:
        index_docset(docset.id, docset.name)


if __name__ == "__main__":
    app()
