import os
import warnings
from typing import Any, Dict, List, Optional, Union

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
            `ARGILLA_API_URL` environment variable or the default http://localhost:6900
            will be used.
        api_key: API Key to connect to the Argilla Server. Defaults to `None`, which
            means that either `ARGILLA_API_KEY` environment variable or the default
            `argilla.apikey` will be used.

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
                `ARGILLA_API_URL` environment variable or the default
                http://localhost:6900 will be used.
            api_key: API Key to connect to the Argilla Server. Defaults to `None`, which
                means that either `ARGILLA_API_KEY` environment variable or the default
                `argilla.apikey` will be used.

        Raises:
            ImportError: if the `argilla` package is not installed.
            ConnectionError: if the connection to Argilla fails.
            FileNotFoundError: if the `FeedbackDataset` retrieval from Argilla fails.
        """

        super().__init__()

        # Import Argilla (not via `import_argilla` to keep hints in IDEs)
        try:
            import argilla as rg  # noqa: F401
        except ImportError:
            raise ImportError(
                "To use the Argilla callback manager you need to have the `argilla` "
                "Python package installed. Please install it with `pip install argilla`"
            )

        # Show a warning message if Argilla will assume the default values will be used
        if api_url is None and os.getenv("ARGILLA_API_URL") is None:
            warnings.warn(
                (
                    "Since `api_url` is None, and the env var `ARGILLA_API_URL` is not"
                    " set, it will default to `http://localhost:6900`."
                ),
            )
        if api_key is None and os.getenv("ARGILLA_API_KEY") is None:
            warnings.warn(
                (
                    "Since `api_key` is None, and the env var `ARGILLA_API_KEY` is not"
                    " set, it will default to `argilla.apikey`."
                ),
            )

        # Connect to Argilla with the provided credentials, if applicable
        try:
            rg.init(
                api_key=api_key,
                api_url=api_url,
            )
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to Argilla with exception: '{e}'.\n"
                "Please check your `api_key` and `api_url`, and make sure that "
                "the Argilla server is up and running. If the problem persists "
                "please report it to https://github.com/argilla-io/argilla/issues "
                "with the label `langchain`."
            ) from e

        # Set the Argilla variables
        self.dataset_name = dataset_name
        self.workspace_name = workspace_name or rg.get_workspace()

        # Retrieve the `FeedbackDataset` from Argilla (without existing records)
        try:
            self.dataset = rg.FeedbackDataset.from_argilla(
                name=self.dataset_name,
                workspace=self.workspace_name,
                with_records=False,
            )
        except Exception as e:
            raise FileNotFoundError(
                "`FeedbackDataset` retrieval from Argilla failed with exception:"
                f" '{e}'.\nPlease check that the dataset with"
                f" name={self.dataset_name} in the"
                f" workspace={self.workspace_name} exists in advance. If you need help"
                " on how to create a `langchain`-compatible `FeedbackDataset` in"
                " Argilla, please visit"
                " https://docs.argilla.io/en/latest/guides/llms/practical_guides/use_argilla_callback_in_langchain.html."  # noqa: E501
                " If the problem persists please report it to"
                " https://github.com/argilla-io/argilla/issues with the label"
                " `langchain`."
            ) from e

        supported_fields = ["prompt", "response"]
        if supported_fields != [field.name for field in self.dataset.fields]:
            raise ValueError(
                f"`FeedbackDataset` with name={self.dataset_name} in the"
                f" workspace={self.workspace_name} "
                "had fields that are not supported yet for the `langchain` integration."
                " Supported fields are: "
                f"{supported_fields}, and the current `FeedbackDataset` fields are"
                f" {[field.name for field in self.dataset.fields]}. "
                "For more information on how to create a `langchain`-compatible"
                " `FeedbackDataset` in Argilla, please visit"
                " https://docs.argilla.io/en/latest/guides/llms/practical_guides/use_argilla_callback_in_langchain.html."  # noqa: E501
            )

        self.prompts: Dict[str, List[str]] = {}

        warnings.warn(
            (
                "The `ArgillaCallbackHandler` is currently in beta and is subject to "
                "change based on updates to `langchain`. Please report any issues to "
                "https://github.com/argilla-io/argilla/issues with the tag `langchain`."
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

        # Push the records to Argilla
        self.dataset.push_to_argilla()

        # Pop current run from `self.runs`
        self.prompts.pop(str(kwargs["run_id"]))

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing when LLM outputs an error."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Do nothing when LLM chain starts."""
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
        """Do nothing when LLM chain ends."""
        prompts = self.prompts[str(kwargs["parent_run_id"] or kwargs["run_id"])]
        if "outputs" in outputs:
            # Creates the records and adds them to the `FeedbackDataset`
            self.dataset.add_records(
                records=[
                    {
                        "fields": {
                            "prompt": prompt,
                            "response": output["text"].strip(),
                        },
                    }
                    for prompt, output in zip(prompts, outputs["outputs"])
                ]
            )
        elif "output" in outputs:
            # Creates the records and adds them to the `FeedbackDataset`
            self.dataset.add_records(
                records=[
                    {
                        "fields": {
                            "prompt": " ".join(prompts),
                            "response": outputs["output"].strip(),
                        },
                    }
                ]
            )
        else:
            raise ValueError(
                "The `outputs` dictionary did not contain the expected keys `outputs` "
                "or `output`."
            )

        # Push the records to Argilla
        self.dataset.push_to_argilla()

        # Pop current run from `self.runs`
        self.prompts.pop(str(kwargs["parent_run_id"] or kwargs["run_id"]))

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
