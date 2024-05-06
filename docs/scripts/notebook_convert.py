import multiprocessing
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Tuple

import nbformat
from nbconvert.exporters import MarkdownExporter

exporter = MarkdownExporter(
    template_name="mdoutput",
    extra_template_basedirs=[
        "/Users/erickfriis/langchain/oss-py/docs/scripts/notebook_convert_templates"
    ],
)
# config = exporter.default_config
# config.TemplateExporter.extra_template_basedirs = [
#     "/Users/erickfriis/langchain/oss-py/docs/scripts/notebook_convert_templates"
# ]
# config.template = "mdoutput"
# config.template_name = "mdoutput"
# exporter.config = config


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

    source_paths_arg = os.environ.get("SOURCE_PATHS")
    source_paths: Iterable[Path]
    if source_paths_arg:
        print("Using SOURCE_PATHS")

        source_path_strs = re.split(r"\s+", source_paths_arg)
        source_paths_stripped = [p.strip() for p in source_path_strs]
        source_paths = [intermediate_docs_dir / p for p in source_paths_stripped if p]
        print(source_paths)
    else:
        source_paths = intermediate_docs_dir.glob("**/*.ipynb")

    with multiprocessing.Pool() as pool:
        pool.map(
            _process_path,
            (
                (notebook_path, intermediate_docs_dir, output_docs_dir)
                for notebook_path in source_paths
            ),
        )
