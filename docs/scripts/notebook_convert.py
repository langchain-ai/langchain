import multiprocessing
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Tuple

import nbformat
from nbconvert.exporters import MarkdownExporter
from nbconvert.preprocessors import Preprocessor


class EscapePreprocessor(Preprocessor):
    def preprocess_cell(self, cell, resources, cell_index):
        if cell.cell_type == "markdown":
            # rewrite .ipynb links to .md
            cell.source = re.sub(
                r"\[([^\]]*)\]\((?![^\)]*//)([^)]*)\.ipynb\)",
                r"[\1](\2.md)",
                cell.source,
            )

        elif cell.cell_type == "code":
            # escape ``` in code
            cell.source = cell.source.replace("```", r"\`\`\`")
            # escape ``` in output

            # allow overriding title based on comment at beginning of cell
            if cell.source.startswith("# title="):
                lines = cell.source.split("\n")
                title = lines[0].split("# title=")[1]
                if title.startswith('"') and title.endswith('"'):
                    title = title[1:-1]
                cell.metadata["title"] = title
                cell.source = "\n".join(lines[1:])

            if "outputs" in cell:
                filter_out = set()
                for i, output in enumerate(cell["outputs"]):
                    if "text" in output:
                        if not output["text"].strip():
                            filter_out.add(i)
                            continue
                        output["text"] = output["text"].replace("```", r"\`\`\`")
                    elif "data" in output:
                        for key, value in output["data"].items():
                            if isinstance(value, str):
                                output["data"][key] = value.replace("```", r"\`\`\`")
                cell["outputs"] = [
                    output
                    for i, output in enumerate(cell["outputs"])
                    if i not in filter_out
                ]

        return cell, resources


class ExtractAttachmentsPreprocessor(Preprocessor):
    """
    Extracts all of the outputs from the notebook file.  The extracted
    outputs are returned in the 'resources' dictionary.
    """

    def preprocess_cell(self, cell, resources, cell_index):
        """
        Apply a transformation on each cell,
        Parameters
        ----------
        cell : NotebookNode cell
            Notebook cell being processed
        resources : dictionary
            Additional resources used in the conversion process.  Allows
            preprocessors to pass variables into the Jinja engine.
        cell_index : int
            Index of the cell being processed (see base.py)
        """

        # Get files directory if it has been specified

        # Make sure outputs key exists
        if not isinstance(resources["outputs"], dict):
            resources["outputs"] = {}

        # Loop through all of the attachments in the cell
        for name, attach in cell.get("attachments", {}).items():
            for mime, data in attach.items():
                if mime not in {
                    "image/png",
                    "image/jpeg",
                    "image/svg+xml",
                    "application/pdf",
                }:
                    continue

                # attachments are pre-rendered. Only replace markdown-formatted
                # images with the following logic
                attach_str = f"({name})"
                if attach_str in cell.source:
                    data = f"(data:{mime};base64,{data})"
                    cell.source = cell.source.replace(attach_str, data)

        return cell, resources


class CustomRegexRemovePreprocessor(Preprocessor):
    def check_conditions(self, cell):
        pattern = re.compile(r"(?s)(?:\s*\Z)|(?:.*#\s*\|\s*output:\s*false.*)")
        rtn = not pattern.match(cell.source)
        if not rtn:
            return False
        else:
            return True

    def preprocess(self, nb, resources):
        nb.cells = [cell for cell in nb.cells if self.check_conditions(cell)]

        return nb, resources


exporter = MarkdownExporter(
    preprocessors=[
        EscapePreprocessor,
        ExtractAttachmentsPreprocessor,
        CustomRegexRemovePreprocessor,
    ],
    template_name="mdoutput",
    extra_template_basedirs=["./scripts/notebook_convert_templates"],
)


def _process_path(tup: Tuple[Path, Path, Path]):
    notebook_path, intermediate_docs_dir, output_docs_dir = tup
    relative = notebook_path.relative_to(intermediate_docs_dir)
    output_path = output_docs_dir / relative.parent / (relative.stem + ".md")
    _convert_notebook(notebook_path, output_path, intermediate_docs_dir)


def _modify_frontmatter(
    body: str, notebook_path: Path, intermediate_docs_dir: Path
) -> str:
    # if frontmatter exists
    rel_path = notebook_path.relative_to(intermediate_docs_dir).as_posix()
    edit_url = (
        f"https://github.com/langchain-ai/langchain/edit/master/docs/docs/{rel_path}"
    )
    frontmatter = {
        "custom_edit_url": edit_url,
    }
    if re.match(r"^[\s\n]*---\n", body):
        # frontmatter already present

        for k, v in frontmatter.items():
            # if key already exists, leave it
            if re.match(f"{k}: ", body):
                continue
            else:
                body = re.sub(r"^[\s\n]*---\n", f"---\n{k}: {v}\n", body, count=1)
        return body
    else:
        insert = "\n".join([f"{k}: {v}" for k, v in frontmatter.items()])
        return f"---\n{insert}\n---\n{body}"


def _convert_notebook(
    notebook_path: Path, output_path: Path, intermediate_docs_dir: Path
) -> Path:
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    body, resources = exporter.from_notebook_node(nb)

    body = _modify_frontmatter(body, notebook_path, intermediate_docs_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(body)

    return output_path


if __name__ == "__main__":
    intermediate_docs_dir = Path(sys.argv[1])
    output_docs_dir = Path(sys.argv[2])

    source_paths_arg = os.environ.get("SOURCE_PATHS")
    source_paths: Iterable[Path]
    if source_paths_arg:
        source_path_strs = re.split(r"\s+", source_paths_arg)
        source_paths_stripped = [p.strip() for p in source_path_strs]
        source_paths = [intermediate_docs_dir / p for p in source_paths_stripped if p]
    else:
        original_paths = list(intermediate_docs_dir.glob("**/*.ipynb"))
        # exclude files that exist in output directory and are newer
        relative_paths = [p.relative_to(intermediate_docs_dir) for p in original_paths]
        out_paths = [
            output_docs_dir / p.parent / (p.stem + ".md") for p in relative_paths
        ]
        source_paths = [
            p
            for p, o in zip(original_paths, out_paths)
            if not o.exists() or o.stat().st_mtime < p.stat().st_mtime
        ]
        print(f"rebuilding {len(source_paths)}/{len(relative_paths)} notebooks")

    with multiprocessing.Pool() as pool:
        pool.map(
            _process_path,
            (
                (notebook_path, intermediate_docs_dir, output_docs_dir)
                for notebook_path in source_paths
            ),
        )
