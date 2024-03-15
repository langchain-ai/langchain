import ast
import glob
import os
from pathlib import Path

CUR_DIR = Path(os.path.abspath(__file__)).parent
PARENT_DIR = CUR_DIR.parent
imports_to_migrate = []
for file_path in glob.glob(str(PARENT_DIR / "libs/langchain/langchain/**/*.py")):
    curr_module = file_path[len(str(PARENT_DIR)) + 16 : -3].replace("/", ".")
    if curr_module[-9:] == ".__init__":
        curr_module = curr_module[:-9]
    with open(file_path, "r") as file:
        raw_contents = file.read()
        module = ast.parse(raw_contents, filename=file_path)
    for node in module.body:
        if not isinstance(
            node,
            ast.ImportFrom,
        ):
            continue
        if "langchain_community" not in node.module:
            continue
        names = [n.__dict__ for n in node.names]
        imports_to_migrate.append(
            {"old_module": curr_module, "new_module": node.module, "imports": names}
        )
    if "def __getattr__" in raw_contents and "from langchain_community" in raw_contents:
        direct_imports = [
            alias.asname or alias.name
            for n in module.body
            if isinstance(n, (ast.Import, ast.ImportFrom))
            for alias in n.names
        ]

        getattr_fn = [
            n
            for n in module.body
            if isinstance(n, ast.FunctionDef) and n.name == "__getattr__"
        ][0]
        community_import = [
            n
            for n in getattr_fn.body
            if isinstance(n, ast.ImportFrom) and "langchain_community" in n.module
        ]
        if not community_import:
            continue
        community_import = community_import[0]
        community_mod = community_import.module + "." + community_import.names[0].name

        all_assign = [
            n
            for n in module.body
            if isinstance(n, ast.Assign) and n.targets[0].id == "__all__"
        ][0]
        all_val = set([e.value for e in all_assign.value.elts])
        indirect_imports = all_val.difference(direct_imports)
        indirect_aliases = [{"name": x, "asname": None} for x in indirect_imports]

        imports_to_migrate.append(
            {
                "old_module": curr_module,
                "new_module": community_mod,
                "imports": indirect_aliases,
            }
        )

old_to_new = {
    (x["old_module"], i["asname"] or i["name"]): {
        "new_module": x["new_module"],
        "name": i["name"],
        "asname": i["asname"],
    }
    for x in imports_to_migrate
    for i in x["imports"]
}

with open(CUR_DIR / "_old_to_new.py", "w") as f:
    f.write(f"OLD_TO_NEW={old_to_new}")
