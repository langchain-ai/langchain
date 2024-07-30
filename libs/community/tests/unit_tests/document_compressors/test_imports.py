from langchain_community.document_compressors import __all__, _module_lookup

EXPECTED_ALL = [
    "LLMLinguaCompressor",
    "OpenVINOReranker",
    "JinaRerank",
    "RankLLMRerank",
    "FlashrankRerank",
    "DashScopeRerank",
    "VolcengineRerank",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
    assert set(__all__) == set(_module_lookup.keys())
