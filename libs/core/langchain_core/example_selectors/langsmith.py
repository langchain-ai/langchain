"""Select examples using a LangSmith few-shot index."""

import json
from typing import Any, Dict, List, Optional, Type, Union

from langsmith import Client
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils

from langchain_core._api import beta
from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.utils.function_calling import convert_to_openai_function


@beta()
class LangSmithExampleSelector(
    BaseExampleSelector, Runnable[Dict[str, Any], Dict[str, Any]]
):
    """Select examples using a LangSmith few-shot index.

    .. dropdown:: Index creation

        .. code-block:: python

            from langchain_core.example_selectors import LangSmithExampleSelector

            examples = [
                {"input": {"question": "..."}, "outputs": {"answer": "..."}}
                ...
            ]
            example_selector = LangSmithExampleSelector(
                k=4,
                dataset_name="foo_bar_task_few_shot_examples",
            )

            example_selector.create_dataset(...)
            example_selector.add_examples(examples)

    .. dropdown:: Retrieving few shot examples

        .. code-block:: python

            from langchain_core.example_selectors import LangSmithExampleSelector
            from langchain_core.prompts import ChatPromptTemplate
            from langchain.chat_models import init_chat_model

            example_selector = LangSmithExampleSelector(
                k=4,
                dataset_name="foo_bar_task_few_shot_examples",
            )

            instructions = "..."
            def construct_prompt(input_: dict) -> list:

                examples = []
                for ex in input_["examples"]:
                    examples.extend([
                        HumanMessage(ex["inputs"]["question"], name="example_user"),
                        AIMessage(ex["outputs"]["answer"], name="example_assistant")
                    ])

                return [
                    SystemMessage(instructions),
                    *examples,
                    HumanMessage(input_["question"])
                ]
            llm = init_chat_model("gpt-4o", temperature=0)

            chain = example_selector | construct_prompt | llm
            chain.invoke({"question": "..."})
    """

    def __init__(
        self,
        *,
        k: int,
        dataset_name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        client: Optional[Client] = None,
        **client_kwargs: Any,
    ) -> None:
        """

        .. note::

            Initializing the ``LangSmithExampleSelector`` does **not** create a dataset.
            This must be done explicitly, either outside the example selector or using
            the ``LangSmithExampleSelector(...).create_dataset(...)`` method.

        Args:
            k: How many examples to return on invocation.
            dataset_name: The name of the dataset of examples.
            dataset_id: The ID of the dataset of examples. Must specify one of
                dataset_name or dataset_id. If both are specified they must correspond
                to the same dataset.
            client: ``langsmith.Client``. If None, then ``client_kwargs`` will be used
                to initialize a new ``langsmith.Client``.
            client_kwargs: If ``client`` isn't specified these keyword args will be
                used ot initialize a new ``langsmith.Client``.
        """
        if client_kwargs and client:
            raise ValueError(
                f"Must specify one and only one of:\n{client=}\n\n{client_kwargs}."
            )
        self.k = k
        self.dataset_name = dataset_name
        self.dataset_id = dataset_id
        self._client = client or Client(**client_kwargs)

    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Retrieve examples from the dataset that are most relevant to the input.

        Args:
            input: A dictionary of input variables.
            config: A config to use when invoking the Runnable.
               The config supports standard keys like 'tags', 'metadata' for tracing
               purposes, 'max_concurrency' for controlling how much work to do
               in parallel, and other keys. Please refer to the
               :class:`~langchain_core.runnables.config.RunnableConfig` for more
               details.
            kwargs: Additional keyword arguments are passed through to
                ``select_examples()``.

        Returns:
            A dictionary with the inputs plus an "examples" key which contains a list
            of the retrieved examples. Each example is a dictionary with top-level
            "inputs" and "outputs" keys which contain a mapping of the input
            and output variables to their example values.
        """
        return {
            **input,
            **{
                "examples": self._call_with_config(
                    self.select_examples,  # type: ignore[arg-type]
                    input,
                    config,
                    **kwargs,
                )
            },
        }

    async def ainvoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Retrieve examples from the dataset that are most relevant to the input.

        Args:
            input: A dictionary of input variables.
            config: A config to use when invoking the Runnable.
               The config supports standard keys like 'tags', 'metadata' for tracing
               purposes, 'max_concurrency' for controlling how much work to do
               in parallel, and other keys. Please refer to the
               :class:`~langchain_core.runnables.config.RunnableConfig` for more
               details.
            kwargs: Additional keyword arguments are passed through to
                ``select_examples()``.

        Returns:
            A dictionary with the inputs plus an "examples" key which contains a list
            of the retrieved examples. Each example is a dictionary with top-level
            "inputs" and "outputs" keys which contain a mapping of the input
            and output variables to their example values.
        """
        return {
            **input,
            **{
                "examples": await self._acall_with_config(
                    self.aselect_examples,  # type: ignore[arg-type]
                    input,
                    config,
                    **kwargs,
                )
            },
        }

    def select_examples(self, input_variables: Dict[str, str]) -> List[Dict[str, Any]]:
        """Select which examples to use based on the inputs.

        Args:
            input_variables: A that maps input variables to their values.

        Returns:
            A list of examples. Each example is a dictionary with a top level "inputs"
            and "outputs" key. The values of these are dictionaries which map the
            input and output variables to their example values.
        """
        search_req_json = json.dumps({"inputs": input_variables, "limit": self.k})
        dataset_id = self._get_dataset_id()
        few_shot_resp = self._client.request_with_retries(
            "POST",
            f"/datasets/{dataset_id}/search",
            headers={**self._client._headers, "Content-Type": "application/json"},
            data=search_req_json,
        )
        ls_utils.raise_for_status_with_text(few_shot_resp)
        return few_shot_resp.json()["examples"]

    def add_examples(self, examples: List[Dict[str, Dict]]) -> Any:
        """Add new examples to store.

        Args:
            examples: A list of dicts. Each dict must have a top-level "inputs" key
                that contains the inputs for the example and an "outputs" key that
                contains the outputs for the example.

        Returns:
            None

        Raises:
            ValueError: if dataset with (self.dataset_name, self.dataset_id) does not
                exist yet.
        """
        if not self.dataset_exists():
            identifiers = ", ".join(
                arg
                for arg in [
                    f"dataset_name={self.dataset_name}",
                    f"dataset_id={self.dataset_id}",
                ]
                if arg
            )
            raise ValueError(
                f"Dataset with {identifiers} does not exist yet."
                f" Please run `LangSmithExampleSelector(...).create_dataset(...)` to"
                f" create this dataset and then try adding examples."
            )
        self._client.create_examples(
            dataset_name=self.dataset_name,
            dataset_id=self.dataset_id,
            inputs=[e["inputs"] for e in examples],
            outputs=[e["outputs"] for e in examples],
        )

    def add_example(self, example: Dict[str, Dict]) -> Any:
        """Add new example to store.

        Args:
            example: A dict that must have a top-level "inputs" key
                that contains the inputs for the example and an "outputs" key that
                contains the outputs for the example.
        """
        return self.add_examples([example])

    def create_dataset(
        self,
        input_schema: Union[Dict, Type],
        output_schema: Union[Dict, Type],
        *,
        description: Optional[str] = None,
    ) -> str:
        """Create a dataset to index examples into and retrieve examples from.

        Args:
            input_schema: The expected schema for all example inputs, passed in as a
                JSON Schema or a TypedDict class.
            output_schema: The expected schema for all example outputs, passed in as a
                JSON Schema or a TypedDict class.
            description: An optional description for the dataset.

        Returns:
            String dataset ID of the newly created dataset. If
            ``LangSmithExampleSelector.dataset_id`` is set, this will be the ID that's
            returned.

        Raises:
            ValueError: If ``LangSmithExampleSelector.dataset_name`` isn't set.
        """
        if not self.dataset_name:
            raise ValueError(
                "`LangSmithExampleSelector.dataset_name` must be set to be able to "
                "create a dataset. Please initialize a new example selector with "
                "init arg `dataset_name` passed in."
            )

        dataset_to_create_json = {
            "name": self.dataset_name,
            "description": description or "Dataset of indexed few-shot examples.",
            "data_type": "kv",
            "inputs_schema_definition": _convert_to_json_schema(input_schema),
            "outputs_schema_definition": _convert_to_json_schema(output_schema),
            "id": self.dataset_id,
        }

        # Create dataset.
        headers = {**self._client._headers, "Content-Type": "application/json"}
        resp = self._client.request_with_retries(
            "POST",
            "/datasets",
            headers=headers,
            data=json.dumps(dataset_to_create_json),
        )

        ls_utils.raise_for_status_with_text(resp)
        dataset = ls_schemas.Dataset(
            **resp.json(),
            _host_url=self._client._host_url,
            _tenant_id=self._client._get_optional_tenant_id(),
        )

        dataset_id = str(dataset.id)

        # Turn on dataset indexing.
        resp = self._client.request_with_retries(
            "POST",
            f"/datasets/{dataset_id}/index",
            headers=headers,
            data=json.dumps({"tag": "latest"}),
        )

        ls_utils.raise_for_status_with_text(resp)

        return dataset_id

    def dataset_exists(self) -> bool:
        """Returns True if the configured dataset already exists, otherwise False."""
        return self._client.has_dataset(
            dataset_name=self.dataset_name, dataset_id=self.dataset_id
        )

    def _get_dataset_id(self) -> str:
        if self.dataset_id:
            return self.dataset_id
        else:
            dataset = self._client.read_dataset(dataset_name=self.dataset_name)
            return str(dataset.id)


def _convert_to_json_schema(schema: Union[Dict, Type]) -> Dict:
    # TODO: Flip so that theres a generic convert_to_json_schema function that
    # convert_to_openai_function uses.
    oai_function = convert_to_openai_function(schema)
    return {
        "title": oai_function["name"],
        "description": oai_function["description"],
        **oai_function["parameters"],
    }
