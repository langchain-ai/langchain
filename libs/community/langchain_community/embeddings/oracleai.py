# Authors:
#   Harichandan Roy (hroy)
#   David Jiang (ddjiang)
#
# -----------------------------------------------------------------------------
# oracleai.py
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import logging
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra

if TYPE_CHECKING:
    from oracledb import Connection

logger = logging.getLogger(__name__)

"""OracleEmbeddings class"""


class OracleEmbeddings(BaseModel, Embeddings):
    """Get Embeddings"""

    """Oracle Connection"""
    conn: Any
    """Embedding Parameters"""
    params: Dict[str, Any]
    """Proxy"""
    proxy: Optional[str] = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    """
    1 - user needs to have create procedure, 
        create mining model, create any directory privilege.
    2 - grant create procedure, create mining model, 
        create any directory to <user>;
    """

    @staticmethod
    def load_onnx_model(
        conn: Connection, dir: str, onnx_file: str, model_name: str
    ) -> None:
        """Load an ONNX model to Oracle Database.
        Args:
            conn: Oracle Connection,
            dir: Oracle Directory,
            onnx_file: ONNX file name,
            model_name: Name of the model.
        """

        try:
            if conn is None or dir is None or onnx_file is None or model_name is None:
                raise Exception("Invalid input")

            cursor = conn.cursor()
            cursor.execute(
                """
                begin
                    dbms_data_mining.drop_model(model_name => :model, force => true);
                    SYS.DBMS_VECTOR.load_onnx_model(:path, :filename, :model, 
                        json('{"function" : "embedding", 
                            "embeddingOutput" : "embedding", 
                            "input": {"input": ["DATA"]}}'));
                end;""",
                path=dir,
                filename=onnx_file,
                model=model_name,
            )

            cursor.close()

        except Exception as ex:
            logger.info(f"An exception occurred :: {ex}")
            traceback.print_exc()
            cursor.close()
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using an OracleEmbeddings.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each input text.
        """

        try:
            import oracledb
        except ImportError as e:
            raise ImportError(
                "Unable to import oracledb, please install with "
                "`pip install -U oracledb`."
            ) from e

        if texts is None:
            return None

        embeddings: List[List[float]] = []
        try:
            # returns strings or bytes instead of a locator
            oracledb.defaults.fetch_lobs = False
            cursor = self.conn.cursor()

            if self.proxy:
                cursor.execute(
                    "begin utl_http.set_proxy(:proxy); end;", proxy=self.proxy
                )

            for text in texts:
                cursor.execute(
                    "select t.* "
                    + "from dbms_vector_chain.utl_to_embeddings(:content, "
                    + "json(:params)) t",
                    content=text,
                    params=json.dumps(self.params),
                )

                for row in cursor:
                    if row is None:
                        embeddings.append([])
                    else:
                        rdata = json.loads(row[0])
                        # dereference string as array
                        vec = json.loads(rdata["embed_vector"])
                        embeddings.append(vec)

            cursor.close()
            return embeddings
        except Exception as ex:
            logger.info(f"An exception occurred :: {ex}")
            traceback.print_exc()
            cursor.close()
            raise

    def embed_query(self, text: str) -> List[float]:
        """Compute query embedding using an OracleEmbeddings.
        Args:
            text: The text to embed.
        Returns:
            Embedding for the text.
        """
        return self.embed_documents([text])[0]


# uncomment the following code block to run the test

"""
# A sample unit test.

''' get the Oracle connection '''
conn = oracledb.connect(
    user="",
    password="",
    dsn="")
print("Oracle connection is established...")

''' params '''
embedder_params = {"provider":"database", "model":"demo_model"}
proxy = ""

''' instance '''
embedder = OracleEmbeddings(conn=conn, params=embedder_params, proxy=proxy)

embed = embedder.embed_query("Hello World!")
print(f"Embedding generated by OracleEmbeddings: {embed}")

conn.close()
print("Connection is closed.")

"""
