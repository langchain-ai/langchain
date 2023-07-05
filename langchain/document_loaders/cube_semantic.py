from typing import List

import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class CubeSemanticLoader(BaseLoader):
    """Load Cube semantic layer metadata."""

    def __init__(
        self,
        cube_api_url: str,
        cube_api_token: str,
    ):
        self.cube_api_url = cube_api_url
        """Use the REST API of your Cube's deployment.
        Please find out more information here:
        https://cube.dev/docs/http-api/rest#configuration-base-path
        """
        self.cube_api_token = cube_api_token
        """Authentication tokens are generated based on your Cube's API secret.
        Please find out more information here:
        https://cube.dev/docs/security#generating-json-web-tokens-jwt
        """

    def load(self) -> List[Document]:
        """Makes a call to Cube's REST API metadata endpoint.

        Returns:
            A list of documents with attributes:
                - page_content=column_name
                - metadata
                    - table_name
                    - column_name
                    - column_data_type
                    - column_title
                    - column_description
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.cube_api_token,
        }

        response = requests.get(self.cube_api_url, headers=headers)
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
                metadata = dict(
                    table_name=str(cube_name),
                    column_name=str(item.get("name")),
                    column_data_type=str(item.get("type")),
                    column_title=str(item.get("title")),
                    column_description=str(item.get("description")),
                )

                page_content = f"table name: {str(cube_name)}, "
                page_content += f"column name: {str(item.get('name'))}, "
                page_content += f"column data type: {str(item.get('type'))}, "
                page_content += f"column title: {str(item.get('title'))}, "
                page_content += f"column description: {str(item.get('description'))}"

                docs.append(Document(page_content=page_content, metadata=metadata))

        return docs
