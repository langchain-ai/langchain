import ast
import glob
import sys
from pathlib import Path
from typing import Union

from _old_to_new import OLD_TO_NEW

OLD_TO_NEW_MODS_ONLY = {k[0] for k in OLD_TO_NEW}


def migrate(dir_: Union[str, Path]) -> None:
    for file_path in glob.glob(str(Path(dir_).absolute() / "**/*.py")):
        with open(file_path, "r") as file:
            lines = file.readlines()
            module = ast.parse("".join(lines), filename=file_path, type_comments=True)

        new_lines = []
        for i, node in enumerate(module.body):
            if not isinstance(node, ast.ImportFrom) or (
                node.module not in OLD_TO_NEW_MODS_ONLY
            ):
                continue
            new_import_froms = []
            for imported in node.names:
                if (node.module, imported.name) in OLD_TO_NEW:
                    new_import = OLD_TO_NEW[(node.module, imported.name)]
                    asname = imported.asname or new_import["asname"]
                    new_alias = ast.alias(name=new_import["name"], as_name=asname)
                    if new_import["new_module"] in [
                        getattr(nb, "module", None) for nb in new_import_froms
                    ]:
                        existing_new_node = [
                            nb
                            for nb in new_import_froms
                            if getattr(nb, "module", None) == new_import["new_module"]
                        ][0]
                        existing_new_node.names.append(new_alias)
                    else:
                        node_params = node.__dict__.copy()
                        node_params.pop("module")
                        node_params.pop("names")
                        new_import_froms.append(
                            ast.ImportFrom(
                                module=new_import["new_module"],
                                names=[new_alias],
                                **node_params,
                            )
                        )
                else:
                    if node.module in [
                        getattr(nb, "module", None) for nb in new_import_froms
                    ]:
                        existing_node = [
                            nb
                            for nb in new_import_froms
                            if getattr(nb, "module", None) == node.module
                        ][0]
                        existing_node.names.append(imported)
                    else:
                        node_params = node.__dict__.copy()
                        node_params["names"] = [imported]
                        new_import_froms.append(ast.ImportFrom(**node_params))
            _str = ast.unparse(
                ast.fix_missing_locations(
                    ast.Module(body=new_import_froms, type_ignores=[])
                )
            )
            _lines = [x + "\n" for x in _str.split("\n")]
            new_lines.append((node.lineno - 1, node.end_lineno, _lines))

        final_lines = []
        last_end = 0
        if new_lines:
            for start, end, _lines in new_lines:
                final_lines += lines[last_end:start] + _lines
                last_end = end
            final_lines += lines[last_end:]
        else:
            final_lines = lines
        with open(file_path, "w") as file:
            file.write("".join(final_lines))


if __name__ == "__main__":
    migrate(sys.argv[1])
