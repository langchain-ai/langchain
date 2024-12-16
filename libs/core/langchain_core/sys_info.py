"""**sys_info** prints information about the system and langchain packages
for debugging purposes.
"""

from collections.abc import Sequence


def _get_sub_deps(packages: Sequence[str]) -> list[str]:
    """Get any specified sub-dependencies."""
    from importlib import metadata

    sub_deps = set()
    _underscored_packages = {pkg.replace("-", "_") for pkg in packages}

    for pkg in packages:
        try:
            required = metadata.requires(pkg)
        except metadata.PackageNotFoundError:
            continue

        if not required:
            continue

        for req in required:
            try:
                cleaned_req = req.split(" ")[0]
            except Exception:  # In case parsing of requirement spec fails
                continue

            if cleaned_req.replace("-", "_") not in _underscored_packages:
                sub_deps.add(cleaned_req)

    return sorted(sub_deps, key=lambda x: x.lower())


def print_sys_info(*, additional_pkgs: Sequence[str] = ()) -> None:
    """Print information about the environment for debugging purposes.

    Args:
        additional_pkgs: Additional packages to include in the output.
    """
    import pkgutil
    import platform
    import sys
    from importlib import metadata, util

    # Packages that do not start with "langchain" prefix.
    other_langchain_packages = [
        "langserve",
        "langsmith",
    ]

    langchain_pkgs = [
        name for _, name, _ in pkgutil.iter_modules() if name.startswith("langchain")
    ]

    langgraph_pkgs = [
        name for _, name, _ in pkgutil.iter_modules() if name.startswith("langgraph")
    ]

    all_packages = sorted(
        set(
            langchain_pkgs
            + langgraph_pkgs
            + other_langchain_packages
            + list(additional_pkgs)
        )
    )

    # Always surface these packages to the top
    order_by = ["langchain_core", "langchain", "langchain_community", "langsmith"]

    for pkg in reversed(order_by):
        if pkg in all_packages:
            all_packages.remove(pkg)
            all_packages = [pkg] + list(all_packages)

    system_info = {
        "OS": platform.system(),
        "OS Version": platform.version(),
        "Python Version": sys.version,
    }
    print()  # noqa: T201
    print("System Information")  # noqa: T201
    print("------------------")  # noqa: T201
    print("> OS: ", system_info["OS"])  # noqa: T201
    print("> OS Version: ", system_info["OS Version"])  # noqa: T201
    print("> Python Version: ", system_info["Python Version"])  # noqa: T201

    # Print out only langchain packages
    print()  # noqa: T201
    print("Package Information")  # noqa: T201
    print("-------------------")  # noqa: T201

    not_installed = []

    for pkg in all_packages:
        try:
            found_package = util.find_spec(pkg)
        except Exception:
            found_package = None
        if found_package is None:
            not_installed.append(pkg)
            continue

        # Package version
        try:
            package_version = metadata.version(pkg)
        except Exception:
            package_version = None

        # Print package with version
        if package_version is not None:
            print(f"> {pkg}: {package_version}")  # noqa: T201
        else:
            print(f"> {pkg}: Installed. No version info available.")  # noqa: T201

    if not_installed:
        print()  # noqa: T201
        print("Optional packages not installed")  # noqa: T201
        print("-------------------------------")  # noqa: T201
        for pkg in not_installed:
            print(f"> {pkg}")  # noqa: T201

    sub_dependencies = _get_sub_deps(all_packages)

    if sub_dependencies:
        print()  # noqa: T201
        print("Other Dependencies")  # noqa: T201
        print("------------------")  # noqa: T201

        for dep in sub_dependencies:
            try:
                dep_version = metadata.version(dep)
                print(f"> {dep}: {dep_version}")  # noqa: T201
            except Exception:
                print(f"> {dep}: Installed. No version info available.")  # noqa: T201


if __name__ == "__main__":
    print_sys_info()
