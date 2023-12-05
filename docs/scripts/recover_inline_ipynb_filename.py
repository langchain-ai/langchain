import json
import pathlib

here = pathlib.Path(__file__).parent.absolute()
json_file = here.parent / ".inline_ipynb.json"

with json_file.open() as f:
    files = json.load(f)

for filename, content in files:
    with open(filename, "w") as f:
        f.write(content)
