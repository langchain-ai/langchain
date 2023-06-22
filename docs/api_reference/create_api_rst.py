import glob
import re

classes = {}
for py in glob.glob("../langchain/**/*.py", recursive=True):
    mod = py[len("../langchain/"):].split(".")[0].replace("/", ".")
    first = mod.split(".")[0]
    if first not in classes:
        classes[first] = []
    with open(py) as f:
        for l in f.readlines():
            found = re.findall(r"^class (.*)\(", l)
            classes[first].extend([mod + "." + c for c in found])

full_doc = """\
.. _api_ref:

=============
API Reference
=============

"""
for mod, clist in sorted(classes.items(), key=lambda kv: kv[0]):
    if not clist:
        continue
    cstring = "\n    ".join(sorted(clist))
    mod_title = mod.replace("_", " ").title()
    if mod_title == "Llms":
        mod_title = mod_title.upper()
    section = f":mod:`langchain.{mod}`: {mod_title}"
    doc = f"""\
{section}
{'=' * (len(section) + 1)}

.. automodule:: langchain.{mod}
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: langchain

.. autosummary::
    :toctree: {mod}
    :template: class.rst

    {cstring}

"""
    full_doc += doc

with open("./api_ref.rst", "w") as f:
    f.write(full_doc)
