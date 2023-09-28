import os
import tempfile
from typing import List, Any

from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.unstructured import UnstructuredBaseLoader
from langchain.schema import Document
import requests
import urllib.parse
from urllib.parse import urljoin
from requests.auth import HTTPBasicAuth


class LakeFSLoader(BaseLoader):
    """Load from `lakeFS`."""

    def __init__(
            self,
            lakefs_access_key: str,
            lakefs_secret_key: str,
            repo: str,
            ref: str = 'main',
            path: str = '',
            lakefs_endpoint: str = 'http://localhost:8000',
    ):
        """

        Args:

        :param lakefs_access_key:
        :param lakefs_secret_key:
        :param lakefs_endpoint:
        :param repo:
        :param ref:
        """

        self.__lakefs_access_key = lakefs_access_key
        self.__lakefs_secret_key = lakefs_secret_key
        self.__lakefs_endpoint = '/'.join([lakefs_endpoint, 'api', 'v1'])
        self.__auth = HTTPBasicAuth(lakefs_access_key, lakefs_secret_key)
        self.repo = repo
        self.ref = ref
        self.path = path

    def load(self) -> List[Document]:
        self.__verify_running_lakefs()
        docs: List[Document] = []
        presigned_urls = self.__ls_object()
        for pu in presigned_urls:
            lakefs_unstructured_loader = UnstructuredLakeFSLoader(pu[1], self.repo, self.ref, pu[0])
            docs.extend(lakefs_unstructured_loader.load())
        return docs

    def __verify_running_lakefs(self):
        version_endpoint = self.__lakefs_endpoint + '/config/version'
        try:
            requests.get(version_endpoint, auth=self.__auth)
        except Exception:
            raise ValueError(
                "lakeFS version couldn't be retrieved. Make sure that your lakeFS server is running."
            )

    def __ls_object(self):
        qp = {'prefix': self.path, 'presign': True}
        eqp = urllib.parse.urlencode(qp)
        objects_ls_endpoint = urljoin(self.__lakefs_endpoint,
                                      f'repositories/{self.repo}/refs/{self.ref}/objects/ls?{eqp}')
        olsr = requests.get(objects_ls_endpoint, auth=self.__auth)
        olsr.raise_for_status()
        olsr = olsr.json()
        return list(
            olsr['results'].map(lambda presigned_url: (presigned_url['path'], presigned_url['physical_address'])))


class UnstructuredLakeFSLoader(UnstructuredBaseLoader):

    def __init__(self, presigned_url: str, repo: str, ref: str = 'main', path: str = '', **unstructured_kwargs: Any):
        """

        Args:

        :param lakefs_access_key:
        :param lakefs_secret_key:
        :param lakefs_endpoint:
        :param repo:
        :param ref:
        """

        super().__init__(**unstructured_kwargs)
        self.presigned_url = presigned_url
        self.repo = repo
        self.ref = ref
        self.path = path

    def _get_metadata(self) -> dict:
        return {'repo': self.repo, 'ref': self.ref}

    def _get_elements(self) -> List:
        from unstructured.partition.auto import partition

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{self.path.split('/')[-1]}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            response = requests.get(self.presigned_url)
            response.raise_for_status()
            with open(file_path, mode="wb") as file:
                file.write(response.content)
            return partition(filename=file_path)
