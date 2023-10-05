import os
import json
from pathlib import Path

DOCS_DIR = Path(os.path.abspath(__file__)).parents[1] / "docs_skeleton" / "docs"


def add_collab_link(cell_content: list, filepath: str) -> list:
    """Inserts the 'Open In Collab' link into the cell content if it doesn't exist."""

    collab_base = (
        "https://colab.research.google.com/github/langchain-ai/langchain/blob/master/"
    )
    collab_link = f"[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)]({collab_base}{filepath})\n"

    if collab_link not in cell_content:
        cell_content = cell_content[:1] + [collab_link] + cell_content[1:]

    return cell_content


def process_directory(directory: str) -> None:
    """Traverses the directory and updates .ipynb files if necessary."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".ipynb"):
                print("Checking", file)
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as ipynb_file:
                    ipynb_data = json.load(ipynb_file)

                try:
                    first_cell_content = ipynb_data["cells"][0]["source"]
                except Exception as e:
                    print("Skipping", filepath, e)
                    continue
                modified_content = add_collab_link(first_cell_content, filepath)
                if modified_content != first_cell_content:
                    print("Inserting link into", filepath)
                    ipynb_data["cells"][0]["source"] = modified_content

                    with open(filepath, "w", encoding="utf-8") as ipynb_file:
                        json.dump(ipynb_data, ipynb_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    process_directory(str(DOCS_DIR))
