from langchain.document_loaders.github import GitHubLoader


def test_integration() -> None:
    title = (
        "ChatOpenAI models don't work with prompts created via ChatPromptTemplate."
        "from_role_strings"
    )
    loader = GitHubLoader(repo="hwchase17/langchain")
    data = loader.load(creator="UmerHA")
    titles = [d.metadata["title"] for d in data]
    assert title in titles
