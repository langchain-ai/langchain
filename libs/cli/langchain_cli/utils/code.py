from typing import List


def split_code(code: str) -> tuple[List[str], List[str], List[str]]:
    """
    Splits the code into 3 blocks, where the import_line and route_line get inserted in the gaps
    """
    lines = code.splitlines()

    split_1 = 0
    split_2 = 0
    for i, line in enumerate(lines):



    rtn = ([], [], [])
    curr_section = 0
    for i, line in enumerate(lines):
        if curr_section == 0:
            # line must match all criteria, otherwise increment to section 1
            stripped_line = line.strip()
            # line must be blank, of format `import x` or `from x import y`
            if not (
                stripped_line == ""
                or stripped_line.startswith("import ")
                or stripped_line.startswith("from ")
                or stripped_line.startswith("#")
            ):
                curr_section = 1
        rtn[curr_section].append(line)
    return rtn


def add_route_code(
    old_code: str, module: str, attr: str, chain_name: str, api_path
) -> str:
    import_line = f"from {module} import {attr} as {chain_name}"
    route_line = f'add_routes(app, {chain_name}, path="{api_path}")'

    beginning, middle, end = split_code(old_code)

    return ""
