import json
import logging
import time
from typing import Iterator, List

import requests
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader

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
        logger.info("Loading dimension values for: %s ...", dimension_name)

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
                logger.error(
                    "Request failed with status code: %s", response.status_code
                )
                break

        if retries == self.dimension_values_max_retries:
            logger.info("Maximum retries reached.")
        return []

    def lazy_load(self) -> Iterator[Document]:
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
                    - cube_data_obj_type
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.cube_api_token,
        }

        logger.info("Loading metadata from %s ...", self.cube_api_url)
        response = requests.get(f"{self.cube_api_url}/meta", headers=headers)
        response.raise_for_status()
        raw_meta_json = response.json()
        cube_data_objects = raw_meta_json.get("cubes", [])

        logger.info("Found %s cube data objects in metadata.", len(cube_data_objects))

        if not cube_data_objects:
            raise ValueError("No cubes found in metadata.")

        for cube_data_obj in cube_data_objects:
            cube_data_obj_name = cube_data_obj.get("name")
            cube_data_obj_type = cube_data_obj.get("type")
            cube_data_obj_is_public = cube_data_obj.get("public")
            measures = cube_data_obj.get("measures", [])
            dimensions = cube_data_obj.get("dimensions", [])

            logger.info("Processing %s ...", cube_data_obj_name)

            if not cube_data_obj_is_public:
                logger.info("Skipping %s because it is not public.", cube_data_obj_name)
                continue

            for item in measures + dimensions:
                column_member_type = "measure" if item in measures else "dimension"
                dimension_values = []
                item_name = str(item.get("name"))
                item_type = str(item.get("type"))

                is_public = bool(item.get("public"))
                if not is_public:
                    logger.info("Skipping %s because it is not public.", item_name)
                    continue

                if (
                    self.load_dimension_values
                    and column_member_type == "dimension"
                    and item_type == "string"
                ):
                    dimension_values = self._get_dimension_values(item_name)

                metadata = dict(
                    table_name=str(cube_data_obj_name),
                    column_name=item_name,
                    column_data_type=item_type,
                    column_title=str(item.get("title")),
                    column_description=str(item.get("description")),
                    column_member_type=column_member_type,
                    column_values=dimension_values,
                    cube_data_obj_type=cube_data_obj_type,
                )

                page_content = f"{str(item.get('title'))}, "
                page_content += f"{str(item.get('description'))}"

                yield Document(page_content=page_content, metadata=metadata)
