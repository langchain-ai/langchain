from typing import Any

from langchain_core.runnables.base import RunnableBindingBase
from langchain_core.runnables.utils import Input, Output


class HubRunnable(RunnableBindingBase[Input, Output]):  # type: ignore[no-redef]
    """An instance of a runnable stored in the LangChain Hub."""

    owner_repo_commit: str

    def __init__(
        self,
        owner_repo_commit: str,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the `HubRunnable`.

        Args:
            owner_repo_commit: The full name of the prompt to pull from in the format of
                `owner/prompt_name:commit_hash` or `owner/prompt_name`
                or just `prompt_name` if it's your own prompt.
            api_url: The URL of the LangChain Hub API.
                Defaults to the hosted API service if you have an api key set,
                or a localhost instance if not.
            api_key: The API key to use to authenticate with the LangChain Hub API.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        from langchain_classic.hub import pull

        pulled = pull(owner_repo_commit, api_url=api_url, api_key=api_key)
        super_kwargs = {
            "kwargs": {},
            "config": {},
            **kwargs,
            "bound": pulled,
            "owner_repo_commit": owner_repo_commit,
        }
        super().__init__(**super_kwargs)
