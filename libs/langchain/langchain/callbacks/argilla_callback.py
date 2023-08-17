import os
import warnings
from typing import Any, Dict, List, Optional, Union

from packaging.version import parse

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult


class ArgillaCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs into Argilla.

    Args:
        dataset_name: name of the `FeedbackDataset` in Argilla. Note that it must
            exist in advance. If you need help on how to create a `FeedbackDataset` in
            Argilla, please visit
            https://docs.argilla.io/en/latest/guides/llms/practical_guides/use_argilla_callback_in_langchain.html.
        workspace_name: name of the workspace in Argilla where the specified
            `FeedbackDataset` lives in. Defaults to `None`, which means that the
            default workspace will be used.
        api_url: URL of the Argilla Server that we want to use, and where the
            `FeedbackDataset` lives in. Defaults to `None`, which means that either
            `ARGILLA_API_URL` environment variable or the default will be used.
        api_key: API Key to connect to the Argilla Server. Defaults to `None`, which
            means that either `ARGILLA_API_KEY` environment variable or the default
            will be used.

    Raises:
        ImportError: if the `argilla` package is not installed.
        ConnectionError: if the connection to Argilla fails.
        FileNotFoundError: if the `FeedbackDataset` retrieval from Argilla fails.

    Examples:
        >>> from langchain.llms import OpenAI
        >>> from langchain.callbacks import ArgillaCallbackHandler
        >>> argilla_callback = ArgillaCallbackHandler(
        ...     dataset_name="my-dataset",
        ...     workspace_name="my-workspace",
        ...     api_url="http://localhost:6900",
        ...     api_key="argilla.apikey",
        ... )
        >>> llm = OpenAI(
        ...     temperature=0,
        ...     callbacks=[argilla_callback],
        ...     verbose=True,
        ...     openai_api_key="API_KEY_HERE",
        ... )
        >>> llm.generate([
        ...     "What is the best NLP-annotation tool out there? (no bias at all)",
        ... ])
        "Argilla, no doubt about it."
    """

    REPO_URL: str = "https://github.com/argilla-io/argilla"
    ISSUES_URL: str = f"{REPO_URL}/issues"
    BLOG_URL: str = "https://docs.argilla.io/en/latest/guides/llms/practical_guides/use_argilla_callback_in_langchain.html"  # noqa: E501

    DEFAULT_API_URL: str = "http://localhost:6900"

    def __init__(
        self,
        dataset_name: str,
        workspace_name: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Initializes the `ArgillaCallbackHandler`.

        Args:
            dataset_name: name of the `FeedbackDataset` in Argilla. Note that it must
                exist in advance. If you need help on how to create a `FeedbackDataset`
                in Argilla, please visit
                https://docs.argilla.io/en/latest/guides/llms/practical_guides/use_argilla_callback_in_langchain.html.
            workspace_name: name of the workspace in Argilla where the specified
                `FeedbackDataset` lives in. Defaults to `None`, which means that the
                default workspace will be used.
            api_url: URL of the Argilla Server that we want to use, and where the
                `FeedbackDataset` lives in. Defaults to `None`, which means that either
                `ARGILLA_API_URL` environment variable or the default will be used.
            api_key: API Key to connect to the Argilla Server. Defaults to `None`, which
                means that either `ARGILLA_API_KEY` environment variable or the default
                will be used.

        Raises:
            ImportError: if the `argilla` package is not installed.
            ConnectionError: if the connection to Argilla fails.
            FileNotFoundError: if the `FeedbackDataset` retrieval from Argilla fails.
        """

        super().__init__()

        # Import Argilla (not via `import_argilla` to keep hints in IDEs)
        try:
            import argilla as rg  # noqa: F401

            self.ARGILLA_VERSION = rg.__version__
        except ImportError:
            raise ImportError(
                "To use the Argilla callback manager you need to have the `argilla` "
                "Python package installed. Please install it with `pip install argilla`"
            )

        # Check whether the Argilla version is compatible
        if parse(self.ARGILLA_VERSION) < parse("1.8.0"):
            raise ImportError(
                f"The installed `argilla` version is {self.ARGILLA_VERSION} but "
                "`ArgillaCallbackHandler` requires at least version 1.8.0. Please "
                "upgrade `argilla` with `pip install --upgrade argilla`."
            )

        # Show a warning message if Argilla will assume the default values will be used
        if api_url is None and os.getenv("ARGILLA_API_URL") is None:
            warnings.warn(
                (
                    "Since `api_url` is None, and the env var `ARGILLA_API_URL` is not"
                    f" set, it will default to `{self.DEFAULT_API_URL}`, which is the"
                    " default API URL in Argilla Quickstart."
                ),
            )
            api_url = self.DEFAULT_API_URL

        if api_key is None and os.getenv("ARGILLA_API_KEY") is None:
            self.DEFAULT_API_KEY = (
                "admin.apikey"
                if parse(self.ARGILLA_VERSION) < parse("1.11.0")
                else "owner.apikey"
            )

            warnings.warn(
                (
                    "Since `api_key` is None, and the env var `ARGILLA_API_KEY` is not"
                    f" set, it will default to `{self.DEFAULT_API_KEY}`, which is the"
                    " default API key in Argilla Quickstart."
                ),
            )
            api_url = self.DEFAULT_API_URL

        # Connect to Argilla with the provided credentials, if applicable
        try:
            rg.init(api_key=api_key, api_url=api_url)
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to Argilla with exception: '{e}'.\n"
                "Please check your `api_key` and `api_url`, and make sure that "
                "the Argilla server is up and running. If the problem persists "
                f"please report it to {self.ISSUES_URL} as an `integration` issue."
            ) from e

        # Set the Argilla variables
        self.dataset_name = dataset_name
        self.workspace_name = workspace_name or rg.get_workspace()

        # Retrieve the `FeedbackDataset` from Argilla (without existing records)
        try:
            extra_args = {}
            if parse(self.ARGILLA_VERSION) < parse("1.14.0"):
                warnings.warn(
                    f"You have Argilla {self.ARGILLA_VERSION}, but Argilla 1.14.0 or"
                    " higher is recommended.",
                    UserWarning,
                )
                extra_args = {"with_records": False}
            self.dataset = rg.FeedbackDataset.from_argilla(
                name=self.dataset_name,
                workspace=self.workspace_name,
                **extra_args,
            )
        except Exception as e:
            raise FileNotFoundError(
                f"`FeedbackDataset` retrieval from Argilla failed with exception `{e}`."
                f"\nPlease check that the dataset with name={self.dataset_name} in the"
                f" workspace={self.workspace_name} exists in advance. If you need help"
                " on how to create a `langchain`-compatible `FeedbackDataset` in"
                f" Argilla, please visit {self.BLOG_URL}. If the problem persists"
                f" please report it to {self.ISSUES_URL} as an `integration` issue."
            ) from e

        supported_fields = ["prompt", "response"]
        if supported_fields != [field.name for field in self.dataset.fields]:
            raise ValueError(
                f"`FeedbackDataset` with name={self.dataset_name} in the workspace="
                f"{self.workspace_name} had fields that are not supported yet for the"
                f"`langchain` integration. Supported fields are: {supported_fields},"
                f" and the current `FeedbackDataset` fields are {[field.name for field in self.dataset.fields]}."  # noqa: E501
                " For more information on how to create a `langchain`-compatible"
                f" `FeedbackDataset` in Argilla, please visit {self.BLOG_URL}."
            )

        self.prompts: Dict[str, List[str]] = {}

        warnings.warn(
            (
                "The `ArgillaCallbackHandler` is currently in beta and is subject to"
                " change based on updates to `langchain`. Please report any issues to"
                f" {self.ISSUES_URL} as an `integration` issue."
            ),
        )

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Save the prompts in memory when an LLM starts."""
        self.prompts.update({str(kwargs["parent_run_id"] or kwargs["run_id"]): prompts})

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Do nothing when a new token is generated."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Log records to Argilla when an LLM ends."""
        # Do nothing if there's a parent_run_id, since we will log the records when
        # the chain ends
        if kwargs["parent_run_id"]:
            return

        # Creates the records and adds them to the `FeedbackDataset`
        prompts = self.prompts[str(kwargs["run_id"])]
        for prompt, generations in zip(prompts, response.generations):
            self.dataset.add_records(
                records=[
                    {
                        "fields": {
                            "prompt": prompt,
                            "response": generation.text.strip(),
                        },
                    }
                    for generation in generations
                ]
            )

        # Pop current run from `self.runs`
        self.prompts.pop(str(kwargs["run_id"]))

        if parse(self.ARGILLA_VERSION) < parse("1.14.0"):
            # Push the records to Argilla
            self.dataset.push_to_argilla()

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing when LLM outputs an error."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """If the key `input` is in `inputs`, then save it in `self.prompts` using
        either the `parent_run_id` or the `run_id` as the key. This is done so that
        we don't log the same input prompt twice, once when the LLM starts and once
        when the chain starts.
        """
        if "input" in inputs:
            self.prompts.update(
                {
                    str(kwargs["parent_run_id"] or kwargs["run_id"]): (
                        inputs["input"]
                        if isinstance(inputs["input"], list)
                        else [inputs["input"]]
                    )
                }
            )

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """If either the `parent_run_id` or the `run_id` is in `self.prompts`, then
        log the outputs to Argilla, and pop the run from `self.prompts`. The behavior
        differs if the output is a list or not.
        """
        if not any(
            key in self.prompts
            for key in [str(kwargs["parent_run_id"]), str(kwargs["run_id"])]
        ):
            return
        prompts = self.prompts.get(str(kwargs["parent_run_id"])) or self.prompts.get(
            str(kwargs["run_id"])
        )
        for chain_output_key, chain_output_val in outputs.items():
            if isinstance(chain_output_val, list):
                # Creates the records and adds them to the `FeedbackDataset`
                self.dataset.add_records(
                    records=[
                        {
                            "fields": {
                                "prompt": prompt,
                                "response": output["text"].strip(),
                            },
                        }
                        for prompt, output in zip(
                            prompts, chain_output_val  # type: ignore
                        )
                    ]
                )
            else:
                # Creates the records and adds them to the `FeedbackDataset`
                self.dataset.add_records(
                    records=[
                        {
                            "fields": {
                                "prompt": " ".join(prompts),  # type: ignore
                                "response": chain_output_val.strip(),
                            },
                        }
                    ]
                )

        # Pop current run from `self.runs`
        if str(kwargs["parent_run_id"]) in self.prompts:
            self.prompts.pop(str(kwargs["parent_run_id"]))
        if str(kwargs["run_id"]) in self.prompts:
            self.prompts.pop(str(kwargs["run_id"]))

        if parse(self.ARGILLA_VERSION) < parse("1.14.0"):
            # Push the records to Argilla
            self.dataset.push_to_argilla()

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing when LLM chain outputs an error."""
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Do nothing when tool starts."""
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Do nothing when agent takes a specific action."""
        pass

    def on_tool_end(
        self,
        output: str,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Do nothing when tool ends."""
        pass

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing when tool outputs an error."""
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Do nothing"""
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Do nothing"""
        pass
