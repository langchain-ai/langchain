import json
import logging
import time
from typing import List

import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class CubeSemanticLoader(BaseLoader):
    """Load `Cube semantic layer` metadata.

    Args:
        cube_api_url: REST API endpoint.
            Use the REST API of your Cube's deployment.
            Please find out more information here:
            https://cube.dev/docs/http-api/rest#configuration-base-path
        cube_api_token: Cube API token.
            Authentication tokens are generated based on your Cube's API secret.
            Please find out more information here:
            https://cube.dev/docs/security#generating-json-web-tokens-jwt
        load_dimension_values: Whether to load dimension values for every string
            dimension or not.
        dimension_values_limit: Maximum number of dimension values to load.
        dimension_values_max_retries: Maximum number of retries to load dimension
            values.
        dimension_values_retry_delay: Delay between retries to load dimension values.
    """

    def __init__(
        self,
        cube_api_url: str,
        cube_api_token: str,
        load_dimension_values: bool = True,
        dimension_values_limit: int = 10_000,
        dimension_values_max_retries: int = 10,
        dimension_values_retry_delay: int = 3,
    ):
        self.cube_api_url = cube_api_url
        self.cube_api_token = cube_api_token
        self.load_dimension_values = load_dimension_values
        self.dimension_values_limit = dimension_values_limit
        self.dimension_values_max_retries = dimension_values_max_retries
        self.dimension_values_retry_delay = dimension_values_retry_delay

    def _get_dimension_values(self, dimension_name: str) -> List[str]:
        """Makes a call to Cube's REST API load endpoint to retrieve
        values for dimensions.

        These values can be used to achieve a more accurate filtering.
        """
        logger.info("Loading dimension values for: {dimension_name}...")

        headers = {
            "Content-Type": "application/json",
            "Authorization": self.cube_api_token,
        }

        query = {
            "query": {
                "dimensions": [dimension_name],
                "limit": self.dimension_values_limit,
            }
        }

        retries = 0
        while retries < self.dimension_values_max_retries:
            response = requests.request(
                "POST",
                f"{self.cube_api_url}/load",
                headers=headers,
                data=json.dumps(query),
            )

            if response.status_code == 200:
                response_data = response.json()
                if (
                    "error" in response_data
                    and response_data["error"] == "Continue wait"
                ):
                    logger.info("Retrying...")
                    retries += 1
                    time.sleep(self.dimension_values_retry_delay)
                    continue
                else:
                    dimension_values = [
                        item[dimension_name] for item in response_data["data"]
                    ]
                    return dimension_values
            else:
                logger.error("Request failed with status code:", response.status_code)
                break

        if retries == self.dimension_values_max_retries:
            logger.info("Maximum retries reached.")
        return []

    def load(self) -> List[Document]:
        """Makes a call to Cube's REST API metadata endpoint.

        Returns:
            A list of documents with attributes:
                - page_content=column_title + column_description
                - metadata
                    - table_name
                    - column_name
                    - column_data_type
                    - column_member_type
                    - column_title
                    - column_description
                    - column_values
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.cube_api_token,
        }

        response = requests.get(f"{self.cube_api_url}/meta", headers=headers)
        response.raise_for_status()
        raw_meta_json = response.json()
        cubes = raw_meta_json.get("cubes", [])
        docs = []

        for cube in cubes:
            if cube.get("type") != "view":
                continue

            cube_name = cube.get("name")

            measures = cube.get("measures", [])
            dimensions = cube.get("dimensions", [])

            for item in measures + dimensions:
                column_member_type = "measure" if item in measures else "dimension"
                dimension_values = []
                item_name = str(item.get("name"))
                item_type = str(item.get("type"))

                if (
                    self.load_dimension_values
                    and column_member_type == "dimension"
                    and item_type == "string"
                ):
                    dimension_values = self._get_dimension_values(item_name)

                metadata = dict(
                    table_name=str(cube_name),
                    column_name=item_name,
                    column_data_type=item_type,
                    column_title=str(item.get("title")),
                    column_description=str(item.get("description")),
                    column_member_type=column_member_type,
                    column_values=dimension_values,
                )

                page_content = f"{str(item.get('title'))}, "
                page_content += f"{str(item.get('description'))}"

                docs.append(Document(page_content=page_content, metadata=metadata))

        return docs
