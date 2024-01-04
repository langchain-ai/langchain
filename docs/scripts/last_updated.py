import glob
from pathlib import Path
import nbformat
import subprocess

_DOCS_DIR = str(Path(__file__).parent.absolute() / ".." / "docs" )


def update_contents(md, last_updated):
    if "\n" not in md:
        md += "\n"
    first_header = md.find("#")
    if first_header >= 0:
        prefix = md[:first_header]
        body = md[first_header:]
    else:
        prefix = ""
        body = md
    return prefix + body.replace("\n", f"\nLast updated: {last_updated}\n", 1)


def last_updated(path):
    process = subprocess.Popen(["git", "log", '-n', '1', '--pretty=format:%cs', path], stdout=subprocess.PIPE)
    output = process.communicate()
    return output[0].decode()


def update_nb(path):
    nb = nbformat.read(path, as_version=4)
    first_md = None
    for cell in nb.cells:
        if cell["cell_type"] == "markdown":
            first_md = cell
            break
    if first_md is None:
        first_md = nbformat.from_dict(
            {"cell_type": "markdown", "source": "", "metadata": {}})
        nb.cells = [first_md] + nb.cells

    _last_updated = last_updated(path)
    first_md.source = update_contents(first_md.source, _last_updated)
    nbformat.validate(nb)
    nbformat.write(nb, path)

def update_md(path):
    with open(path, "r") as f:
        md = f.read()
    _last_updated = last_updated(path)
    md = update_contents(md, _last_updated)
    with open(path, "w") as f:
        f.write(md)


for path in glob.glob(_DOCS_DIR + "/**/*.*", recursive=True):
    if Path(path).suffix == ".ipynb":
        update_nb(path)
    elif Path(path).suffix.startswith(".md"):
        update_md(path)