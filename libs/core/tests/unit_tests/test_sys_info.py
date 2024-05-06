from langchain_core.sys_info import print_sys_info


def test_print_sys_info() -> None:
    """Super simple test to that no exceptions are triggered when calling code."""
    print_sys_info()
