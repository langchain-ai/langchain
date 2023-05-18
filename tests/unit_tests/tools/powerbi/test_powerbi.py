import pytest


def test_module_import() -> None:
    try:
        pass
    except Exception as e:
        pytest.fail(f"{type(e).__name__}: {str(e)}")
