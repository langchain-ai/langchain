from typing import Any, Dict, List, Optional, Union

from langchain_core.utils import get_from_dict_or_env

from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.graph_store import GraphStore
from langchain_community.utilities.vertexai import get_client_info


class GoogleEnterpriseKnowledgeGraph(GraphStore):
    """Google Cloud Enterrprise Knowledge Graph wrapper.

    Parameters:
    project_id (str): The Google Cloud Project ID.
    location (str): The Google Cloud location for the Knowledge Graph. Default: global
    """

    def __init__(self, project_id: str, location: str = "global") -> None:
        """Create a new Enterprise Knowledge Graph wrapper instance."""
        try:
            from google.cloud import enterpriseknowledgegraph as ekg
        except ImportError:
            raise ValueError(
                "Could not import enterpriseknowledgegraph python package. "
                "Please install it with `pip install google-cloud-enterpriseknowledgegraph`."
            )

        project_id = get_from_dict_or_env(
            {"project_id": project_id}, "project_id", "PROJECT_ID"
        )
        location = get_from_dict_or_env({"location": location}, "location", "LOCATION")

        self._client = ekg.EnterpriseKnowledgeGraphServiceClient(
            client_info=get_client_info(module="enterprise-knowledge-graph")
        )
        self._parent = self._client.common_location_path(
            project=project_id, location=location
        )

    @property
    def get_schema(self) -> str:
        """Returns the schema of the Graph"""
        raise NotImplementedError

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        """Returns the schema of the Graph database"""
        raise NotImplementedError

    def refresh_schema(self) -> None:
        """Refreshes the graph schema information."""
        raise NotImplementedError

    def add_graph_documents(
        self, graph_documents: List[GraphDocument], include_source: bool = False
    ) -> None:
        """Take GraphDocument as input as uses it to construct a graph."""
        raise NotImplementedError

    def query(
        self, query: str, params: Optional[Dict[str, Any]] = None  # type: ignore[override]
    ) -> List[Dict[str, Any]]:
        """Query Enterprise Knowledge Graph.

        Args:
            query (str): The search query.
            params (Dict): Parameters to send with search request.
                Options:
                    - languages (Sequence[str]) - Sequence of ISO 639-1 Codes to return.
                    - types (Sequence[str]) - Sequence of schema.org types to return.
                    - limit (int) - Number of entities to return.
        """
        from google.cloud import enterpriseknowledgegraph as ekg

        params = params or {}
        languages = params.get("languages", None)
        types = params.get("types", None)
        limit = params.get("limit", None)

        response = self._client.search(
            request=ekg.SearchRequest(
                parent=self._parent,
                query=query,
                languages=languages,
                types=types,
                limit=limit,
            )
        )
        results: List[Dict[str, Union[str, List, Dict]]] = []

        for item in response.item_list_element:
            result = item.get("result", {})
            image = result.get("image", {})
            detailed_description = result.get("detailedDescription", {})

            identifiers = {}
            for identifier in result.get("identifier", []):
                identifiers[identifier.get("name")] = identifier.get("value")

            results.append(
                {
                    "name": result.get("name"),
                    "description": result.get("description"),
                    "url": result.get("url"),
                    "types": result.get("@type"),
                    "cloud_mid": result.get("@id"),
                    "image": {
                        "url": image.get("url"),
                        "content_url": image.get("content_url"),
                    },
                    "detailed_description": {
                        "article_body": detailed_description.get("articleBody"),
                        "url": detailed_description.get("url"),
                        "license": detailed_description.get("license"),
                    },
                    "identifiers": identifiers,
                }
            )
        return results
