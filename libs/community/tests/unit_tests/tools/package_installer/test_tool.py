from langchain_community.tools.package_installer.tool import (
    PackageInstallInput,
    PackageInstallTool,
)


def test_package_install_input_single() -> None:
    """
    Test single package input.
    """
    input_obj = PackageInstallInput(package_names="pandas")
    assert (
        input_obj.package_names == "pandas"
    ), "Package names do not match for single package"


def test_package_install_input_multiple() -> None:
    """
    Test multiple package input.
    """
    input_obj = PackageInstallInput(package_names=["pandas", "numpy"])
    assert input_obj.package_names == [
        "pandas",
        "numpy",
    ], "Package names do not match for multiple packages"


def test_install_single_package() -> None:
    """
    Test the installation of a single package.
    """
    install_tool = PackageInstallTool().as_tool()
    result = install_tool.run({"package_names": "pandas"})
    assert result, "Installation failed"


def test_install_multiple_packages() -> None:
    """
    Test the installation of multiple packages.
    """
    install_tool = PackageInstallTool().as_tool()
    result = install_tool.run({"package_names": ["pandas", "numpy"]})
    assert result, "Installation failed"
