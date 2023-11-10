import glob
import os
from pathlib import Path
import re
import shutil


TEMPLATES_DIR = Path(os.path.abspath(__file__)).parents[2] / "templates"
DOCS_TEMPLATES_DIR = Path(os.path.abspath(__file__)).parents[1] / "docs" / "templates"


# os.mkdir(DOCS_TEMPLATES_DIR)

readmes = list(glob.glob(str(TEMPLATES_DIR) + "/*/README.md"))
destinations = [readme[len(str(TEMPLATES_DIR)) + 1:-10] + ".md" for readme in readmes]
for source, destination in zip(readmes, destinations):
    full_destination = DOCS_TEMPLATES_DIR / destination
    shutil.copyfile(source, full_destination)
    with open(full_destination, "r") as f:
        content = f.read()
    # remove images
    content = re.sub("\!\[.*?\]\((.*?)\)", "", content)
    with open(full_destination, "w") as f:
        f.write(content)

sidebar_hidden = """---
sidebar_class_name: hidden
---

"""
TEMPLATES_INDEX_DESTINATION = DOCS_TEMPLATES_DIR / "index.md"
# shutil.copyfile(TEMPLATES_DIR / "docs" / "INDEX.MD", TEMPLATES_INDEX_DESTINATION)
with open(TEMPLATES_INDEX_DESTINATION, "r") as f:
    contents = f.read()
with open(TEMPLATES_INDEX_DESTINATION, "w") as f:
    f.write(sidebar_hidden + contents)

