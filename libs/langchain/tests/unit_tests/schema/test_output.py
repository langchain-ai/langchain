from langchain_classic.schema.output import __all__

EXPECTED_ALL = [
    "ChatGeneration",
    "ChatGenerationChunk",
    "ChatResult",
    "Generation",
    "GenerationChunk",
    "LLMResult",
    "RunInfo",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
