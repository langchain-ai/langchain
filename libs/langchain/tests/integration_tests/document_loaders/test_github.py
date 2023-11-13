from langchain.document_loaders.github import GitHubIssuesLoader


def test_issues_load() -> None:
    title = "DocumentLoader for GitHub"
    loader = GitHubIssuesLoader(
        repo="langchain-ai/langchain", creator="UmerHA", state="all"
    )
    docs = loader.load()
    titles = [d.metadata["title"] for d in docs]
    assert title in titles
    assert all(doc.metadata["creator"] == "UmerHA" for doc in docs)
