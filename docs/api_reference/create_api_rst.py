import glob
import re

members = {}
for py in glob.glob("../langchain/**/*.py", recursive=True):
    mod = py[len("../langchain/"):].split(".")[0].replace("/", ".")
    top_level = mod.split(".")[0]
    if top_level not in members:
        members[top_level] = {"classes": [], "functions": []}
    with open(py) as f:
        for l in f.readlines():
            cls = re.findall(r"^class ([^_].*)\(", l)
            members[top_level]["classes"].extend([mod + "." + c for c in cls])
            func = re.findall(r"^def ([^_].*)\(", l)
            members[top_level]["functions"].extend([mod + "." + f for f in func])

full_doc = """\
.. _api_ref:

=============
API Reference
=============

"""
for mod, _members in sorted(members.items(), key=lambda kv: kv[0]):
    classes = _members["classes"]
    functions = _members["functions"]
    if not (classes or functions):
        continue

    mod_title = mod.replace("_", " ").title()
    if mod_title == "Llms":
        mod_title = mod_title.upper()
    section = f":mod:`langchain.{mod}`: {mod_title}"
    full_doc += f"""\
{section}
{'=' * (len(section) + 1)}

.. automodule:: langchain.{mod}
    :no-members:
    :no-inherited-members:

"""

    if classes:
        cstring = "\n    ".join(sorted(classes))
        full_doc += f"""\
Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: {mod}
    :template: class.rst

    {cstring}

"""
    if functions:
        fstring = "\n    ".join(sorted(functions))
        full_doc += f"""\
Functions
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: {mod}

    {fstring}

"""

with open("./api_ref.rst", "w") as f:
    f.write(full_doc)
