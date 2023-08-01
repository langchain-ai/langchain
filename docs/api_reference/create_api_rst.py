"""Script for auto-generating api_reference.rst"""
import glob
import re
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2].absolute()
PKG_DIR = ROOT_DIR / "libs" / "langchain" / "langchain"
EXP_DIR = ROOT_DIR / "libs" / "experimental" / "langchain_experimental"
WRITE_FILE = Path(__file__).parent / "api_reference.rst"
EXP_WRITE_FILE = Path(__file__).parent / "experimental_api_reference.rst"


def load_members(dir: Path) -> dict:
    members: dict = {}
    for py in glob.glob(str(dir) + "/**/*.py", recursive=True):
        module = py[len(str(dir)) + 1 :].replace(".py", "").replace("/", ".")
        top_level = module.split(".")[0]
        if top_level not in members:
            members[top_level] = {"classes": [], "functions": []}
        with open(py, "r") as f:
            for line in f.readlines():
                cls = re.findall(r"^class ([^_].*)\(", line)
                members[top_level]["classes"].extend([module + "." + c for c in cls])
                func = re.findall(r"^def ([^_].*)\(", line)
                afunc = re.findall(r"^async def ([^_].*)\(", line)
                func_strings = [module + "." + f for f in func + afunc]
                members[top_level]["functions"].extend(func_strings)
    return members


def construct_doc(pkg: str, members: dict) -> str:
    full_doc = f"""\
=============
``{pkg}`` API Reference
=============

"""
    for module, _members in sorted(members.items(), key=lambda kv: kv[0]):
        classes = _members["classes"]
        functions = _members["functions"]
        if not (classes or functions):
            continue
        section = f":mod:`{pkg}.{module}`"
        full_doc += f"""\
{section}
{'=' * (len(section) + 1)}

.. automodule:: {pkg}.{module}
    :no-members:
    :no-inherited-members:

"""

        if classes:
            cstring = "\n    ".join(sorted(classes))
            full_doc += f"""\
Classes
--------------
.. currentmodule:: {pkg}

.. autosummary::
    :toctree: {module}
    :template: class.rst

    {cstring}

"""
        if functions:
            fstring = "\n    ".join(sorted(functions))
            full_doc += f"""\
Functions
--------------
.. currentmodule:: {pkg}

.. autosummary::
    :toctree: {module}
    :template: function.rst

    {fstring}

"""
    return full_doc


def main() -> None:
    lc_members = load_members(PKG_DIR)
    lc_doc = ".. _api_reference:\n\n" + construct_doc("langchain", lc_members)
    with open(WRITE_FILE, "w") as f:
        f.write(lc_doc)
    exp_members = load_members(EXP_DIR)
    exp_doc = ".. _experimental_api_reference:\n\n" + construct_doc("langchain_experimental", exp_members)
    with open(EXP_WRITE_FILE, "w") as f:
        f.write(exp_doc)


if __name__ == "__main__":
    main()
