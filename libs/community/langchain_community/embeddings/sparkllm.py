import base64
import hashlib
import hmac
import json
import logging
from datetime import datetime
from time import mktime
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

import numpy as np
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from numpy import ndarray

# SparkLLMTextEmbeddings is an embedding model provided by iFLYTEK Co., Ltd.. (https://iflytek.com/en/).

# Official Website: https://www.xfyun.cn/doc/spark/Embedding_api.html
# Developers need to create an application in the console first, use the appid, APIKey,
# and APISecret provided in the application for authentication,
# and generate an authentication URL for handshake.
# You can get one by registering at https://console.xfyun.cn/services/bm3.
# SparkLLMTextEmbeddings support 2K token window and preduces vectors with
# 2560 dimensions.

logger = logging.getLogger(__name__)


class Url:
    def __init__(self, host: str, path: str, schema: str) -> None:
        self.host = host
        self.path = path
        self.schema = schema
        pass


class SparkLLMTextEmbeddings(BaseModel, Embeddings):
    """SparkLLM Text Embedding models.

    To use, you should have the environment variable "SPARK_APP_ID","SPARK_API_KEY"
    and "SPARK_API_SECRET" set your APP_ID, API_KEY and API_SECRET or pass it
    as a name parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import SparkLLMTextEmbeddings

            embeddings = SparkLLMTextEmbeddings(
                spark_app_id="your-app-id",
                spark_api_key="your-api-key",
                spark_api_secret="your-api-secret"
            )
            text = "This is a test query."
            query_result = embeddings.embed_query(text)

    """

    spark_app_id: Optional[SecretStr] = Field(default=None, alias="app_id")
    """Automatically inferred from env var `SPARK_APP_ID` if not provided."""
    spark_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Automatically inferred from env var `SPARK_API_KEY` if not provided."""
    spark_api_secret: Optional[SecretStr] = Field(default=None, alias="api_secret")
    """Automatically inferred from env var `SPARK_API_SECRET` if not provided."""
    base_url: str = Field(default="https://emb-cn-huabei-1.xf-yun.com/")
    """Base URL path for API requests"""
    domain: Literal["para", "query"] = Field(default="para")
    """This parameter is used for which Embedding this time belongs to.
    If "para"(default), it belongs to document Embedding. 
    If "query", it belongs to query Embedding."""

    class Config:
        """Configuration for this pydantic object"""

        allow_population_by_field_name = True

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that auth token exists in environment."""
        values["spark_app_id"] = convert_to_secret_str(
            get_from_dict_or_env(values, "spark_app_id", "SPARK_APP_ID")
        )
        values["spark_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "spark_api_key", "SPARK_API_KEY")
        )
        values["spark_api_secret"] = convert_to_secret_str(
            get_from_dict_or_env(values, "spark_api_secret", "SPARK_API_SECRET")
        )
        return values

    def _embed(self, texts: List[str], host: str) -> Optional[List[List[float]]]:
        """Internal method to call Spark Embedding API and return embeddings.

        Args:
            texts: A list of texts to embed.
            host: Base URL path for API requests

        Returns:
            A list of list of floats representing the embeddings,
            or list with value None if an error occurs.
        """
        app_id = ""
        api_key = ""
        api_secret = ""
        if self.spark_app_id:
            app_id = self.spark_app_id.get_secret_value()
        if self.spark_api_key:
            api_key = self.spark_api_key.get_secret_value()
        if self.spark_api_secret:
            api_secret = self.spark_api_secret.get_secret_value()
        url = self._assemble_ws_auth_url(
            request_url=host,
            method="POST",
            api_key=api_key,
            api_secret=api_secret,
        )
        embed_result: list = []
        for text in texts:
            query_context = {"messages": [{"content": text, "role": "user"}]}
            content = self._get_body(app_id, query_context)
            response = requests.post(
                url, json=content, headers={"content-type": "application/json"}
            ).text
            res_arr = self._parser_message(response)
            if res_arr is not None:
                embed_result.append(res_arr.tolist())
            else:
                embed_result.append(None)
        return embed_result

    def embed_documents(self, texts: List[str]) -> Optional[List[List[float]]]:  # type: ignore[override]
        """Public method to get embeddings for a list of documents.

        Args:
            texts: The list of texts to embed.

        Returns:
            A list of embeddings, one for each text, or None if an error occurs.
        """
        return self._embed(texts, self.base_url)

    def embed_query(self, text: str) -> Optional[List[float]]:  # type: ignore[override]
        """Public method to get embedding for a single query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text, or None if an error occurs.
        """
        result = self._embed([text], self.base_url)
        return result[0] if result is not None else None

    @staticmethod
    def _assemble_ws_auth_url(
        request_url: str, method: str = "GET", api_key: str = "", api_secret: str = ""
    ) -> str:
        u = SparkLLMTextEmbeddings._parse_url(request_url)
        host = u.host
        path = u.path
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        signature_origin = "host: {}\ndate: {}\n{} {} HTTP/1.1".format(
            host, date, method, path
        )
        signature_sha = hmac.new(
            api_secret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        signature_sha_str = base64.b64encode(signature_sha).decode(encoding="utf-8")
        authorization_origin = (
            'api_key="%s", algorithm="%s", headers="%s", signature="%s"'
            % (api_key, "hmac-sha256", "host date request-line", signature_sha_str)
        )
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode(
            encoding="utf-8"
        )
        values = {"host": host, "date": date, "authorization": authorization}

        return request_url + "?" + urlencode(values)

    @staticmethod
    def _parse_url(request_url: str) -> Url:
        stidx = request_url.index("://")
        host = request_url[stidx + 3 :]
        schema = request_url[: stidx + 3]
        edidx = host.index("/")
        if edidx <= 0:
            raise AssembleHeaderException("invalid request url:" + request_url)
        path = host[edidx:]
        host = host[:edidx]
        u = Url(host, path, schema)
        return u

    def _get_body(self, appid: str, text: dict) -> Dict[str, Any]:
        body = {
            "header": {"app_id": appid, "uid": "39769795890", "status": 3},
            "parameter": {
                "emb": {"domain": self.domain, "feature": {"encoding": "utf8"}}
            },
            "payload": {
                "messages": {
                    "text": base64.b64encode(json.dumps(text).encode("utf-8")).decode()
                }
            },
        }
        return body

    @staticmethod
    def _parser_message(
        message: str,
    ) -> Optional[ndarray]:
        data = json.loads(message)
        code = data["header"]["code"]
        if code != 0:
            logger.warning(f"Request error: {code}, {data}")
            return None
        else:
            text_base = data["payload"]["feature"]["text"]
            text_data = base64.b64decode(text_base)
            dt = np.dtype(np.float32)
            dt = dt.newbyteorder("<")
            text = np.frombuffer(text_data, dtype=dt)
            if len(text) > 2560:
                array = text[:2560]
            else:
                array = text
            return array


class AssembleHeaderException(Exception):
    """Exception raised for errors in the header assembly."""

    def __init__(self, msg: str) -> None:
        self.message = msg
