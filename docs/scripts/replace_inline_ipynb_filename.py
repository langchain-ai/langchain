import re
import json
import pathlib

here = pathlib.Path(__file__).parent.absolute()
json_file = here.parent / ".inline_ipynb.json"
url_pattern = re.compile('(\./.*)\.ipynb(.*)')

doc = pathlib.Path("docs")
files = []

for file in doc.rglob("*.ipynb"):
    if ".ipynb_checkpoints" in file.as_posix():
        continue
    with file.open() as f:
        for line in f:
            if ".ipynb" in line:
                if (match := url_pattern.search(line)):
                    f.seek(0)
                    files.append([file, f.read()])

with json_file.open("w") as f:
    json.dump([[file.as_posix(), content] for file, content in files], f)

for file, content in files:
    content = re.sub(url_pattern, r"\1\2", content)
    with file.open("w") as f:
        f.write(content)


