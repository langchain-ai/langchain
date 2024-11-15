try:
    import tomllib
except ImportError:
    # if prior to 3.11, use alternative toml library
    import toml as tomllib


def test_cli_template_version() -> None:
    """
    Confirm that the version in the CLI pyproject file is the same as the version in the package.
    """
