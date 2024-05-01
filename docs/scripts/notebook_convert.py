import sys
from pathlib import Path

import nbformat
from nbconvert.exporters import MarkdownExporter
import multiprocessing
from typing import Tuple

exporter = MarkdownExporter()


def _process_path(tup: Tuple[Path, Path, Path]):
    notebook_path, intermediate_docs_dir, output_docs_dir = tup
    relative = notebook_path.relative_to(intermediate_docs_dir)
    output_path = output_docs_dir / relative.parent / (relative.stem + ".md")
    print(notebook_path)
    _convert_notebook(notebook_path, output_path)


def _convert_notebook(notebook_path: Path, output_path: Path):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    body, resources = exporter.from_notebook_node(nb)

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    with open(output_path, "w") as f:
        f.write(body)

    return output_path


if __name__ == "__main__":
    intermediate_docs_dir = Path(sys.argv[1])
    output_docs_dir = Path(sys.argv[2])

    with multiprocessing.Pool() as pool:
        pool.map(
            _process_path,
            (
                (notebook_path, intermediate_docs_dir, output_docs_dir)
                for notebook_path in intermediate_docs_dir.glob("**/*.ipynb")
            ),
        )
