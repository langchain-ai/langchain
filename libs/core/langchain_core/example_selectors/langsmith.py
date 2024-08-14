"""Select examples using a LangSmith few-shot index."""

import json
from typing import Any, Dict, List, Optional

from langsmith import Client
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils

from langchain_core._api import beta
from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.runnables import Runnable, RunnableConfig


@beta()
class LangSmithExampleSelector(BaseExampleSelector, Runnable[Dict, List[Dict]]):
    """Select examples using a LangSmith few-shot index.

    .. dropdown:: Index creation

        .. code-block:: python

            from langchain_core.example_selectors import LangSmithExampleSelector

            examples = [
                ...
            ]
            example_selector = LangSmithExampleSelector(
                dataset_name="foo_bar_task_few_shot_examples",
                ...
            )

            example_selector.create_dataset(...)
            example_selector.add_examples(examples)

    .. dropdown:: Retrieving few shot examples

        .. code-block:: python

            from langchain_core.example_selectors import LangSmithExampleSelector
            from langchain_core.prompts import ChatPromptTemplate
            from langchain.chat_models import init_chat_model

            example_selector = LangSmithExampleSelector(
                dataset_name="foo_bar_task_few_shot_examples",
                ...
            )
            example_prompt = ChatPromptTemplate(
                [("human", "{{inputs.question}}"), ("ai", "{{outputs.answer}}")],
                template_format="mustache",
            )
            prompt = ChatPromptTemplate([
                ("system", "..."),
                ("examples", example_prompt),
                ("human", "{question}"),
            ])
            llm = init_chat_model("gpt-4o", temperature=0)

            chain = example_selector | prompt | llm

            ...

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
        """Initialize selector.

        .. note::

            Initializing the ``LangSmithExampleSelector`` does **not** create a dataset.
            This must be done explicitly, either outside the example selector or using
            the ``LangSmithExampleSelector(...).create_dataset(...)`` method.

        Args:
            k: ...
            dataset_name: ...
            dataset_id: ...
            client: ...
            client_kwargs: ...
        """
        self.k = k
        self.dataset_name = dataset_name
        self.dataset_id = dataset_id
        self._client = client or Client(**client_kwargs)

    def add_example(self, example: Dict[str, Dict]) -> Any:
        """Add new example to store.

        Args:
            example: A dict that must have a top-level "inputs" key
                that contains the inputs for the example and an "outputs" key that
                contains the outputs for the example.
        """
        return self.add_examples([example])

    def add_examples(self, examples: List[Dict[str, Dict]]) -> Any:
        """Add new examples to store.

        Args:
            examples: A list of dicts. Each dict must have a top-level "inputs" key
                that contains the inputs for the example and an "outputs" key that
                contains the outputs for the example.

        Returns:
            ...
        """
        if not self.has_dataset():
            identifiers = ", ".join(
                arg
                for arg in [
                    f"dataset_name={self.dataset_name}",
                    f"dataset_id={self.dataset_id}",
                ]
                if arg
            )
            raise ValueError(
                f"Dataset with {identifiers} doesn't exist yet."
                f" Please run `LangSmithExampleSelector(...).create_dataset(...)`"
            )
        self._client.create_examples(
            dataset_name=self.dataset_name,
            dataset_id=self.dataset_id,
            inputs=[e["inputs"] for e in examples],
            outputs=[e["outputs"] for e in examples],
        )

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs.

        Args:
            input_variables: A that maps input variables to their values.

        Returns:
            ...
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
        return few_shot_resp.json()

    def create_dataset(
        self,
        input_schema: Dict,
        output_schema: Dict,
        *,
        description: Optional[str] = None,
        data_type: str = "kv",
    ) -> str:
        """Create a dataset to index examples into and retrieve examples from.

        Args:
            input_schema:
            output_schema:
            description:
            data_type:

        Returns:
            String dataset ID of the newly created dataset.
        """
        dataset_to_create_json = {
            "name": self.dataset_name,
            "description": description or "...",
            "data_type": data_type,
            "inputs_schema_definition": input_schema,
            "outputs_schema_definition": output_schema,
            "id": self.dataset_id,
        }

        resp = self._client.request_with_retries(
            "POST",
            "/datasets",
            headers={**self._client._headers, "Content-Type": "application/json"},
            data=json.dumps(dataset_to_create_json),
        )

        ls_utils.raise_for_status_with_text(resp)
        dataset = ls_schemas.Dataset(
            **resp.json(),
            _host_url=self._client._host_url,
            _tenant_id=self._client._get_optional_tenant_id(),
        )

        return str(dataset.id)

    def has_dataset(self) -> bool:
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

    def invoke(
        self, input: Dict[str, Any], config: Optional[RunnableConfig] = None
    ) -> Dict:
        return {
            **input,
            **{
                "examples": self._call_with_config(
                    self.select_examples,
                    input,
                    config,
                    run_type="example_selector",
                )
            },
        }

    async def ainvoke(
        self, input: Dict[str, Any], config: Optional[RunnableConfig] = None
    ) -> Dict:
        return {
            **input,
            **{
                "examples": await self._acall_with_config(
                    self.aselect_examples,
                    input,
                    config,
                    run_type="example_selector",
                )
            },
        }
