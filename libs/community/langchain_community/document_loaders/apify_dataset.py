from typing import Any, Callable, Dict, List

from langchain_core._api import deprecated
from langchain_core.documents import Document
from pydantic import BaseModel, model_validator

from langchain_community.document_loaders.base import BaseLoader


@deprecated(
    since="0.3.18",
    message=(
        "This class is deprecated and will be removed in a future version. "
        "You can swap to using the `ApifyDatasetLoader`"
        " implementation in `langchain_apify` package. "
        "See <https://github.com/apify/langchain-apify>"
    ),
    alternative_import="langchain_apify.ApifyDatasetLoader",
)
class ApifyDatasetLoader(BaseLoader, BaseModel):
    """Load datasets from `Apify` web scraping, crawling, and data extraction platform.

    For details, see https://docs.apify.com/platform/integrations/langchain

    Example:
        .. code-block:: python

            from langchain_community.document_loaders import ApifyDatasetLoader
            from langchain_core.documents import Document

            loader = ApifyDatasetLoader(
                dataset_id="YOUR-DATASET-ID",
                dataset_mapping_function=lambda dataset_item: Document(
                    page_content=dataset_item["text"], metadata={"source": dataset_item["url"]}
                ),
            )
            documents = loader.load()
    """  # noqa: E501

    apify_client: Any
    """An instance of the ApifyClient class from the apify-client Python package."""
    dataset_id: str
    """The ID of the dataset on the Apify platform."""
    dataset_mapping_function: Callable[[Dict], Document]
    """A custom function that takes a single dictionary (an Apify dataset item)
     and converts it to an instance of the Document class."""

    def __init__(
        self, dataset_id: str, dataset_mapping_function: Callable[[Dict], Document]
    ):
        """Initialize the loader with an Apify dataset ID and a mapping function.

        Args:
            dataset_id (str): The ID of the dataset on the Apify platform.
            dataset_mapping_function (Callable): A function that takes a single
                dictionary (an Apify dataset item) and converts it to an instance
                of the Document class.
        """
        super().__init__(
            dataset_id=dataset_id, dataset_mapping_function=dataset_mapping_function
        )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate environment.

        Args:
            values: The values to validate.
        """

        try:
            from apify_client import ApifyClient

            client = ApifyClient()
            if httpx_client := getattr(client.http_client, "httpx_client"):
                httpx_client.headers["user-agent"] += "; Origin/langchain"

            values["apify_client"] = client
        except ImportError:
            raise ImportError(
                "Could not import apify-client Python package. "
                "Please install it with `pip install apify-client`."
            )

        return values

    def load(self) -> List[Document]:
        """Load documents."""
        dataset_items = (
            self.apify_client.dataset(self.dataset_id).list_items(clean=True).items
        )
        return list(map(self.dataset_mapping_function, dataset_items))
