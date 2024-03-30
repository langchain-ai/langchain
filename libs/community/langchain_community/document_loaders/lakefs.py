import os
import tempfile
import urllib.parse
from typing import Any, List, Optional
from urllib.parse import urljoin

import requests
from langchain_core.documents import Document
from requests.auth import HTTPBasicAuth

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.unstructured import UnstructuredBaseLoader


class LakeFSClient:
    """Client for lakeFS."""

    def __init__(
        self,
        lakefs_access_key: str,
        lakefs_secret_key: str,
        lakefs_endpoint: str,
    ):
        self.__endpoint = "/".join([lakefs_endpoint, "api", "v1/"])
        self.__auth = HTTPBasicAuth(lakefs_access_key, lakefs_secret_key)
        try:
            health_check = requests.get(
                urljoin(self.__endpoint, "healthcheck"), auth=self.__auth
            )
            health_check.raise_for_status()
        except Exception:
            raise ValueError(
                "lakeFS server isn't accessible. Make sure lakeFS is running."
            )

    def ls_objects(
        self, repo: str, ref: str, path: str, presign: Optional[bool]
    ) -> List:
        qp = {"prefix": path, "presign": presign}
        eqp = urllib.parse.urlencode(qp)
        objects_ls_endpoint = urljoin(
            self.__endpoint, f"repositories/{repo}/refs/{ref}/objects/ls?{eqp}"
        )
        olsr = requests.get(objects_ls_endpoint, auth=self.__auth)
        olsr.raise_for_status()
        olsr_json = olsr.json()
        return list(
            map(
                lambda res: (res["path"], res["physical_address"]), olsr_json["results"]
            )
        )

    def is_presign_supported(self) -> bool:
        config_endpoint = self.__endpoint + "config"
        response = requests.get(config_endpoint, auth=self.__auth)
        response.raise_for_status()
        config = response.json()
        return config["storage_config"]["pre_sign_support"]


class LakeFSLoader(BaseLoader):
    """Load from `lakeFS`."""

    repo: str
    ref: str
    path: str

    def __init__(
        self,
        lakefs_access_key: str,
        lakefs_secret_key: str,
        lakefs_endpoint: str,
        repo: Optional[str] = None,
        ref: Optional[str] = "main",
        path: Optional[str] = "",
    ):
        """

        :param lakefs_access_key: [required] lakeFS server's access key
        :param lakefs_secret_key: [required] lakeFS server's secret key
        :param lakefs_endpoint: [required] lakeFS server's endpoint address,
               ex: https://example.my-lakefs.com
        :param repo: [optional, default = ''] target repository
        :param ref: [optional, default = 'main'] target ref (branch name,
               tag, or commit ID)
        :param path: [optional, default = ''] target path
        """

        self.__lakefs_client = LakeFSClient(
            lakefs_access_key, lakefs_secret_key, lakefs_endpoint
        )
        self.repo = "" if repo is None or repo == "" else str(repo)
        self.ref = "main" if ref is None or ref == "" else str(ref)
        self.path = "" if path is None else str(path)

    def set_path(self, path: str) -> None:
        self.path = path

    def set_ref(self, ref: str) -> None:
        self.ref = ref

    def set_repo(self, repo: str) -> None:
        self.repo = repo

    def load(self) -> List[Document]:
        self.__validate_instance()
        presigned = self.__lakefs_client.is_presign_supported()
        docs: List[Document] = []
        objs = self.__lakefs_client.ls_objects(
            repo=self.repo, ref=self.ref, path=self.path, presign=presigned
        )
        for obj in objs:
            lakefs_unstructured_loader = UnstructuredLakeFSLoader(
                obj[1], self.repo, self.ref, obj[0], presigned
            )
            docs.extend(lakefs_unstructured_loader.load())
        return docs

    def __validate_instance(self) -> None:
        if self.repo is None or self.repo == "":
            raise ValueError(
                "no repository was provided. use `set_repo` to specify a repository"
            )
        if self.ref is None or self.ref == "":
            raise ValueError("no ref was provided. use `set_ref` to specify a ref")
        if self.path is None:
            raise ValueError("no path was provided. use `set_path` to specify a path")


class UnstructuredLakeFSLoader(UnstructuredBaseLoader):
    """Load from `lakeFS` as unstructured data."""

    def __init__(
        self,
        url: str,
        repo: str,
        ref: str = "main",
        path: str = "",
        presign: bool = True,
        **unstructured_kwargs: Any,
    ):
        """Initialize UnstructuredLakeFSLoader.

        Args:

        :param lakefs_access_key:
        :param lakefs_secret_key:
        :param lakefs_endpoint:
        :param repo:
        :param ref:
        """

        super().__init__(**unstructured_kwargs)
        self.url = url
        self.repo = repo
        self.ref = ref
        self.path = path
        self.presign = presign

    def _get_metadata(self) -> dict:
        return {"repo": self.repo, "ref": self.ref, "path": self.path}

    def _get_elements(self) -> List:
        from unstructured.partition.auto import partition

        local_prefix = "local://"

        if self.presign:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = f"{temp_dir}/{self.path.split('/')[-1]}"
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                response = requests.get(self.url)
                response.raise_for_status()
                with open(file_path, mode="wb") as file:
                    file.write(response.content)
                return partition(filename=file_path)
        elif not self.url.startswith(local_prefix):
            raise ValueError(
                "Non pre-signed URLs are supported only with 'local' blockstore"
            )
        else:
            local_path = self.url[len(local_prefix) :]
            return partition(filename=local_path)
