# based on outerbounds/nbdoc

from nbdev.export import nbglob
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import Preprocessor
from pathlib import Path
import re


class WriteTitle(Preprocessor):
    """Modify the code-fence with the filename upon %%writefile cell magic."""

    pattern = r"(^[\S\s]*%%writefile\s)(\S+)\n"

    def preprocess_cell(self, cell, resources, index):
        print("here")
        m = re.match(self.pattern, cell.source)
        if m:
            filename = m.group(2)
            ext = filename.split(".")[-1]
            cell.metadata.magics_language = f'{ext} title="{filename}"'
            cell.metadata.script = True
            cell.metadata.file_ext = ext
            cell.metadata.filename = filename
            cell.outputs = []
        return cell, resources


def get_exporter():
    # c = Config()
    # c.MarkdownExporter.preprocessors = [WriteTitle]
    exporter = MarkdownExporter(
        config={"MarkdownExporter": {"preprocessors": [WriteTitle]}}
    )
    return exporter


def process_file(fname: Path, force: bool = False) -> None:
    fname_rel = fname.relative_to(basedir)
    fname_out_ipynb = outdir / fname_rel
    fname_out = fname_out_ipynb.with_suffix(".md")

    if (
        force
        or not fname_out.exists()
        or fname.stat().st_mtime > fname_out.stat().st_mtime
    ):
        print(f"Converting {fname_rel} to markdown")
        exporter = get_exporter()
        output, _ = exporter.from_filename(fname)
        fname_out.write_text(output)
        print(fname_out)


if __name__ == "__main__":
    # parallel process
    basedir = Path(__file__).parent.parent / "docs"
    outdir = Path(__file__).parent.parent.parent.parent / "_dist" / "docs"
    files = nbglob(basedir, recursive=True)

    fname = files[0]
    process_file(fname, True)

    # for fname in files:
    #     process_file(fname)

    # print(fname_out)
    # for fname in files:
    #     fname_out = fname.with_suffix('.md')
