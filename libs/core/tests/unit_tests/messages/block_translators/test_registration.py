import pkgutil
from pathlib import Path

import pytest

from langchain_core.messages.block_translators import PROVIDER_TRANSLATORS


def test_all_providers_registered() -> None:
    """Test that all block translators implemented in langchain-core are registered.

    If this test fails, it is likely that a block translator is implemented but not
    registered on import. Check that the provider is included in
    ``langchain_core.messages.block_translators.__init__._register_translators``.
    """
    package_path = (
        Path(__file__).parents[4] / "langchain_core" / "messages" / "block_translators"
    )

    for module_info in pkgutil.iter_modules([str(package_path)]):
        module_name = module_info.name

        # Skip the __init__ module, any private modules, and ``langchain_v0``, which is
        # only used to parse v0 multimodal inputs.
        if module_name.startswith("_") or module_name == "langchain_v0":
            continue

        if module_name not in PROVIDER_TRANSLATORS:
            pytest.fail(f"Block translator not registered: {module_name}")
