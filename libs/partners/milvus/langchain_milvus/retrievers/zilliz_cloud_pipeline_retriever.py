from typing import Any, Dict, List, Optional

import requests
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class ZillizCloudPipelineRetriever(BaseRetriever):
    """`Zilliz Cloud Pipeline` retriever.

    Parameters:
        pipeline_ids: A dictionary of pipeline ids.
            Valid keys: "ingestion", "search", "deletion".
        token: Zilliz Cloud's token. Defaults to "".
        cloud_region: The region of Zilliz Cloud's cluster.
            Defaults to 'gcp-us-west1'.
    """

    pipeline_ids: Dict
    token: str = ""
    cloud_region: str = "gcp-us-west1"

    def _get_relevant_documents(
        self,
        query: str,
        top_k: int = 10,
        offset: int = 0,
        output_fields: List = [],
        filter: str = "",
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Get documents relevant to a query.

        Args:
            query: String to find relevant documents for
            top_k: The number of results. Defaults to 10.
            offset: The number of records to skip in the search result.
                Defaults to 0.
            output_fields: The extra fields to present in output.
            filter: The Milvus expression to filter search results.
                Defaults to "".
            run_manager: The callbacks handler to use.

        Returns:
            List of relevant documents
        """
        if "search" in self.pipeline_ids:
            search_pipe_id = self.pipeline_ids.get("search")
        else:
            raise Exception(
                "A search pipeline id must be provided in pipeline_ids to "
                "get relevant documents."
            )
        domain = (
            f"https://controller.api.{self.cloud_region}.zillizcloud.com/v1/pipelines"
        )
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        url = f"{domain}/{search_pipe_id}/run"

        params = {
            "data": {"query_text": query},
            "params": {
                "limit": top_k,
                "offset": offset,
                "outputFields": output_fields,
                "filter": filter,
            },
        }

        response = requests.post(url, headers=headers, json=params)
        if response.status_code != 200:
            raise RuntimeError(response.text)
        response_dict = response.json()
        if response_dict["code"] != 200:
            raise RuntimeError(response_dict)
        response_data = response_dict["data"]
        search_results = response_data["result"]
        return [
            Document(
                page_content=result.pop("text")
                if "text" in result
                else result.pop("chunk_text"),
                metadata=result,
            )
            for result in search_results
        ]

    def add_texts(
        self, texts: List[str], metadata: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """
        Add documents to store.
        Only supported by a text ingestion pipeline in Zilliz Cloud.

        Args:
            texts: A list of text strings.
            metadata: A key-value dictionary of metadata will
                be inserted as preserved fields required by ingestion pipeline.
                Defaults to None.
        """
        if "ingestion" in self.pipeline_ids:
            ingeset_pipe_id = self.pipeline_ids.get("ingestion")
        else:
            raise Exception(
                "An ingestion pipeline id must be provided in pipeline_ids to"
                " add documents."
            )
        domain = (
            f"https://controller.api.{self.cloud_region}.zillizcloud.com/v1/pipelines"
        )
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        url = f"{domain}/{ingeset_pipe_id}/run"

        metadata = {} if metadata is None else metadata
        params = {"data": {"text_list": texts}}
        params["data"].update(metadata)

        response = requests.post(url, headers=headers, json=params)
        if response.status_code != 200:
            raise Exception(response.text)
        response_dict = response.json()
        if response_dict["code"] != 200:
            raise Exception(response_dict)
        response_data = response_dict["data"]
        return response_data

    def add_doc_url(
        self, doc_url: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """
        Add a document from url.
        Only supported by a document ingestion pipeline in Zilliz Cloud.

        Args:
            doc_url: A document url.
            metadata: A key-value dictionary of metadata will
                be inserted as preserved fields required by ingestion pipeline.
                Defaults to None.
        """
        if "ingestion" in self.pipeline_ids:
            ingest_pipe_id = self.pipeline_ids.get("ingestion")
        else:
            raise Exception(
                "An ingestion pipeline id must be provided in pipeline_ids to "
                "add documents."
            )
        domain = (
            f"https://controller.api.{self.cloud_region}.zillizcloud.com/v1/pipelines"
        )
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        url = f"{domain}/{ingest_pipe_id}/run"

        params = {"data": {"doc_url": doc_url}}
        metadata = {} if metadata is None else metadata
        params["data"].update(metadata)

        response = requests.post(url, headers=headers, json=params)
        if response.status_code != 200:
            raise Exception(response.text)
        response_dict = response.json()
        if response_dict["code"] != 200:
            raise Exception(response_dict)
        response_data = response_dict["data"]
        return response_data

    def delete(self, key: str, value: Any) -> Dict:
        """
        Delete documents. Only supported by a deletion pipeline in Zilliz Cloud.

        Args:
            key: input name to run the deletion pipeline
            value: input value to run deletion pipeline
        """
        if "deletion" in self.pipeline_ids:
            deletion_pipe_id = self.pipeline_ids.get("deletion")
        else:
            raise Exception(
                "A deletion pipeline id must be provided in pipeline_ids to "
                "add documents."
            )
        domain = (
            f"https://controller.api.{self.cloud_region}.zillizcloud.com/v1/pipelines"
        )
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        url = f"{domain}/{deletion_pipe_id}/run"

        params = {"data": {key: value}}

        response = requests.post(url, headers=headers, json=params)
        if response.status_code != 200:
            raise Exception(response.text)
        response_dict = response.json()
        if response_dict["code"] != 200:
            raise Exception(response_dict)
        response_data = response_dict["data"]
        return response_data
