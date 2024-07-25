import importlib
from pathlib import Path

PKG_ROOT = Path(__file__).parent.parent.parent
PKG_CODE = PKG_ROOT / "langchain_experimental"


def test_importable_all() -> None:
    """Test that all modules in langchain_experimental are importable."""
    failures = []
    found_at_least_one = False
    for path in PKG_CODE.rglob("*.py"):
        relative_path = str(Path(path).relative_to(PKG_CODE)).replace("/", ".")
        if relative_path.endswith(".typed"):
            continue
        if relative_path.endswith("/__init__.py"):
            # Then strip __init__.py
            s = "/__init__.py"
            module_name = relative_path[: -len(s)]
        else:  # just strip .py
            module_name = relative_path[:-3]

        if not module_name:
            continue
        try:
            module = importlib.import_module("langchain_experimental." + module_name)
        except ImportError:
            failures.append("langchain_experimental." + module_name)
            continue

        all_ = getattr(module, "__all__", [])
        for cls_ in all_:
            try:
                getattr(module, cls_)
            except AttributeError:
                failures.append(f"{module_name}.{cls_}")

        found_at_least_one = True

    if failures:
        raise AssertionError(
            "The following modules or classes could not be imported: "
            + ", ".join(failures)
        )

    assert found_at_least_one is True
