from typing import Any, Optional

from langchain_core.runnables.base import Input, Output, RunnableBindingBase


class HubRunnable(RunnableBindingBase[Input, Output]):
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
        **kwargs: Any,
    ) -> None:
        from langchain.hub import pull

        pulled = pull(owner_repo_commit, api_url=api_url, api_key=api_key)
        super_kwargs = {
            "kwargs": {},
            "config": {},
            **kwargs,
            "bound": pulled,
            "owner_repo_commit": owner_repo_commit,
        }
        super().__init__(**super_kwargs)
