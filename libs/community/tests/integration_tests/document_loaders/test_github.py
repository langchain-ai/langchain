from langchain_community.document_loaders.github import GitHubIssuesLoader


def test_issues_load() -> None:
    title = " Add caching to BaseChatModel (issue #1644)"
    loader = GitHubIssuesLoader(
        repo="langchain-ai/langchain",
        creator="UmerHA",
        state="all",
        per_page=3,
        page=2,
        access_token="""""",
    )
    docs = loader.load()
    titles = [d.metadata["title"] for d in docs]
    assert title in titles
    assert all(doc.metadata["creator"] == "UmerHA" for doc in docs)
    assert len(docs) == 3
