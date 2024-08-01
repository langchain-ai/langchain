import sys
import tomllib

if __name__ == "__main__":
    # Get the TOML file path from the command line argument
    toml_file = sys.argv[1]

    # read toml file
    with open(toml_file, "rb") as file:
        toml_data = tomllib.load(file)

    # see if we're releasing an rc
    version = toml_data["tool"]["poetry"]["version"]
    releasing_rc = "rc" in version

    # if not, iterate through dependencies and make sure none allow prereleases
    if not releasing_rc:
        dependencies = toml_data["tool"]["poetry"]["dependencies"]
        for lib in dependencies:
            dep_version = dependencies[lib]
            dep_version_string = (
                dep_version["version"] if isinstance(dep_version, dict) else dep_version
            )

            if "rc" in dep_version_string:
                raise ValueError(
                    f"Dependency {lib} has a prerelease version. Please remove this."
                )

            if isinstance(dep_version, dict) and dep_version.get(
                "allow-prereleases", False
            ):
                raise ValueError(
                    f"Dependency {lib} has allow-prereleases set to true. Please remove this."
                )
