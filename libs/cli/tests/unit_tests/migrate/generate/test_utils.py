from langchain_cli.namespaces.migrate.generate.utils import PKGS_ROOT


def test_root() -> None:
    assert PKGS_ROOT.name == "libs"
