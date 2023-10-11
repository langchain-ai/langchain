from typing import Optional

from langchain.schema.runnable.base import Input, Output, RunnableBinding


class HubRunnable(RunnableBinding[Input, Output]):
    """
    An instance of a runnable stored in the LangChain Hub.
    """

    owner_repo_commit: str

    def __init__(
        self,
        owner_repo_commit: str,
        *,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        from langchain.hub import pull

        self.owner_repo_commit = owner_repo_commit
        pulled = pull(owner_repo_commit, api_url=api_url, api_key=api_key)
        super().__init__(bound=pulled, kwargs={}, config={})
