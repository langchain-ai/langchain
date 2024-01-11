"""Print information about the system and langchain packages for debugging purposes."""
from typing import Sequence


def print_sys_info(*, additional_pkgs: Sequence[str] = tuple()) -> None:
    """Print information about the environment for debugging purposes."""
    import platform
    import sys
    from importlib import metadata, util

    packages = [
        "langchain_core",
        "langchain",
        "langchain_community",
        "langserve",
    ] + list(additional_pkgs)

    system_info = {
        "OS": platform.system(),
        "OS Version": platform.version(),
        "Python Version": sys.version,
    }
    print()
    print("System Information")
    print("------------------")
    print("> OS: ", system_info["OS"])
    print("> OS Version: ", system_info["OS Version"])
    print("> Python Version: ", system_info["Python Version"])

    # Print out only langchain packages
    print()
    print("Package Information")
    print("-------------------")

    for pkg in packages:
        try:
            found_package = util.find_spec(pkg)
        except Exception:
            found_package = None
        if found_package is None:
            print(f"> {pkg}: Not Found")
            continue

        # Package version
        try:
            package_version = metadata.version(pkg)
        except Exception:
            package_version = None

        # Print package with version
        if package_version is not None:
            print(f"> {pkg}: {package_version}")
        else:
            print(f"> {pkg}: Found")


if __name__ == "__main__":
    print_sys_info()
