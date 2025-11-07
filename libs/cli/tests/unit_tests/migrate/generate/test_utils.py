from langchain_cli.namespaces.migrate.generate.utils import PKGS_ROOT


def test_root() -> None:
    if PKGS_ROOT.name != "libs":
        msg = "Expected PKGS_ROOT.name to be 'libs'."
        raise ValueError(msg)
