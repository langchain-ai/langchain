from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.agent_toolkits.github.toolkit import (
        BranchName,
        CommentOnIssue,
        CreateFile,
        CreatePR,
        CreateReviewRequest,
        DeleteFile,
        DirectoryPath,
        GetIssue,
        GetPR,
        GitHubToolkit,
        NoInput,
        ReadFile,
        SearchCode,
        SearchIssuesAndPRs,
        UpdateFile,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "NoInput": "langchain_community.agent_toolkits.github.toolkit",
    "GetIssue": "langchain_community.agent_toolkits.github.toolkit",
    "CommentOnIssue": "langchain_community.agent_toolkits.github.toolkit",
    "GetPR": "langchain_community.agent_toolkits.github.toolkit",
    "CreatePR": "langchain_community.agent_toolkits.github.toolkit",
    "CreateFile": "langchain_community.agent_toolkits.github.toolkit",
    "ReadFile": "langchain_community.agent_toolkits.github.toolkit",
    "UpdateFile": "langchain_community.agent_toolkits.github.toolkit",
    "DeleteFile": "langchain_community.agent_toolkits.github.toolkit",
    "DirectoryPath": "langchain_community.agent_toolkits.github.toolkit",
    "BranchName": "langchain_community.agent_toolkits.github.toolkit",
    "SearchCode": "langchain_community.agent_toolkits.github.toolkit",
    "CreateReviewRequest": "langchain_community.agent_toolkits.github.toolkit",
    "SearchIssuesAndPRs": "langchain_community.agent_toolkits.github.toolkit",
    "GitHubToolkit": "langchain_community.agent_toolkits.github.toolkit",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "BranchName",
    "CommentOnIssue",
    "CreateFile",
    "CreatePR",
    "CreateReviewRequest",
    "DeleteFile",
    "DirectoryPath",
    "GetIssue",
    "GetPR",
    "GitHubToolkit",
    "NoInput",
    "ReadFile",
    "SearchCode",
    "SearchIssuesAndPRs",
    "UpdateFile",
]
