import sys
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
        raise Exception(
            "The workspace corresponding to the provided DOCUGAMI_API_KEY does not have any docsets."
        )

    docsets = docsets_response.docsets
    if all_docsets:
        selected_docsets = [d for d in docsets]
    elif docset_id:
        if docset_id not in [d.id for d in docsets]:
            raise Exception(f"Error: Docset with ID {docset_id} does not exist.")
        selected_docsets = [d for d in docsets if d.id == docset_id]
    else:
        typer.echo("Your workspace contains the following Docsets:\n")
        for idx, docset in enumerate(docsets, start=1):
            print(f"{idx}: {docset.name} (ID: {docset.id})")
        user_input = typer.prompt(
            "\nPlease enter the number(s) of the docset(s) to index (comma-separated) or 'all' in index all docsets"
        )

        if user_input.lower() == "all":
            selected_docsets = [d for d in docsets]
        else:
            selected_indices = [int(i.strip()) for i in user_input.split(",")]
            selected_docsets = [
                docsets[idx - 1] for idx in selected_indices if 0 < idx <= len(docsets)
            ]

    for docset in [d for d in selected_docsets if d is not None]:
        if not docset.id or not docset.name:
            raise Exception(f"Docset must have ID as well as Name: {docset}")

        index_docset(docset.id, docset.name)


if __name__ == "__main__":
    if sys.gettrace():
        # This code will only run if a debugger is attached
        main(docset_id="s79br3gqd0g6")
    else:
        app()
