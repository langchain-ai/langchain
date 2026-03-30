from langchain_a2a import __all__

EXPECTED_ALL = [
    "MultiAgentA2AClient",
    "get_tools",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
