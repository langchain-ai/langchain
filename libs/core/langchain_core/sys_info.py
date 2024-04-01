"""**sys_info** prints information about the system and langchain packages
for debugging purposes.
"""

from typing import Sequence


def print_sys_info(*, additional_pkgs: Sequence[str] = tuple()) -> None:
    """Print information about the environment for debugging purposes."""
    import pkgutil
    import platform
    import sys
    from importlib import metadata, util

    # Packages that do not start with "langchain" prefix.
    other_langchain_packages = [
        "langserve",
        "langgraph",
        "langsmith",
    ]

    langchain_pkgs = [
        name for _, name, _ in pkgutil.iter_modules() if name.startswith("langchain")
    ]

    all_packages = sorted(
        set(langchain_pkgs + other_langchain_packages + list(additional_pkgs))
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
        print("Packages not installed (Not Necessarily a Problem)")  # noqa: T201
        print("--------------------------------------------------")  # noqa: T201
        print("The following packages were not found:")  # noqa: T201
        print()  # noqa: T201
        for pkg in not_installed:
            print(f"> {pkg}")  # noqa: T201


if __name__ == "__main__":
    print_sys_info()
