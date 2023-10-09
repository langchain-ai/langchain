from typing import List
import requests
import logging
import json

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class DbtSemanticLayerLoader(BaseLoader):
    """Load dbt semantic layer metadata."""

    def __init__(
        self,
        environment_id: str,
        service_token: str,
        hostname: str = "cloud.getdbt.com",
        load_dimension_values: bool = True,
        dimension_values_max_retries: int = 10,
        dimension_values_retry_delay: int = 3,
    ):
        """Learn how to setup the Semantic Layer here:
        https://docs.getdbt.com/docs/use-dbt-semantic-layer/setup-sl

        - environment_id: dbt Cloud environment id
        - TODO
        """
        self._environment_id = environment_id
        self._load_dimension_values = load_dimension_values
        self._semantic_layer_url = f"https://{hostname}/semantic-layer/api/graphql"
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {service_token}",
        }

    '''
    def _get_dimension_values(self, dimension_name: str, metric_name: str) -> List[str]:
        """Makes a call to dbt's Semantic Layer GraphQL API to retrieve
        values for dimensions.
        """
        logger.info("Loading dimension values for: {dimension_name}...")

        gql_query = """mutation GetDimensionValuesQuery(
            $environmentId: BigInt!,
            $groupBy: [String!]!,
            $metrics: [String!]
        ) {
            createDimensionValuesQuery(
                environmentId: $environmentId,
                groupBy: $groupBy,
                metrics: $metrics
            ) {
                queryId
            }
        }"""
        post_data = {
            "query": gql_query,
            "variables": {
                "environmentId": self._environment_id,
                "groupBy": dimension_name,
            },
        }
        response = requests.post(
            self._semantic_layer_url,
            headers=self._headers,
            data=json.dump(post_data),
        )
        query_id = response.get("query_id")

        retries = 0
        while retries < self.dimension_values_max_retries:
            response = requests.post(
                self._semantic_layer_url,
                headers=self._headers,
                data=json.dump(create_gql_query),
            )

            if response.status_code == 200 and response.json():
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
        '''

    def load(self) -> List[Document]:
        """Makes a call to the dbt Semantic Layer GraphQL API.

        Returns:
            A list of documents with attributes:
                - page_content=column_name
                - metadata
                    - table_name
                    - column_name
                    - column_description
                    - column_values
        """
        gql_query = """query GetMetrics($environmentId: BigInt!) {
            metrics(environmentId: $environmentId) {
                name
                description
                dimensions {
                    name
                    description
                }
            }
        }"""
        post_data = {
            "query": gql_query,
            "variables": {"environmentId": self._environment_id},
        }
        response = requests.post(
            self._semantic_layer_url, headers=self._headers, data=json.dump(post_data)
        )
        response.raise_for_status()
        metrics = response.json().get("data", {}).get("metrics", [])
        docs = []
        for metric in metrics:
            for dimension in metric.get("dimensions", []):
                dimension_values = []
                #if self._load_dimension_values:
                #    dimension_values = self._get_dimension_values(dimension.get("name"))

                metadata = dict(
                    metric_name=str(metric.get("name")),
                    metric_description=str(metric.get("description")),
                    dimension_name=str(dimension.get("name")),
                    dimension_description=str(dimension.get("description")),
                    dimension_values=dimension_values,
                )

                page_content = f"{str(dimension.get('name'))}, "
                page_content += f"{str(dimension.get('description'))}"

                docs.append(Document(page_content=page_content, metadata=metadata))
        return docs
