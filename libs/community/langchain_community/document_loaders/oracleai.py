# Authors:
#   Harichandan Roy (hroy)
#   David Jiang (ddjiang)
#
# -----------------------------------------------------------------------------
# oracleai.py
# -----------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import struct
import time
import traceback
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

if TYPE_CHECKING:
    from oracledb import Connection

logger = logging.getLogger(__name__)

"""ParseOracleDocMetadata class"""


class ParseOracleDocMetadata(HTMLParser):
    """Parse Oracle doc metadata..."""

    def __init__(self) -> None:
        super().__init__()
        self.reset()
        self.match = False
        self.metadata: Dict[str, Any] = {}

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "meta":
            entry: Optional[str] = ""
            for name, value in attrs:
                if name == "name":
                    entry = value
                if name == "content":
                    if entry:
                        self.metadata[entry] = value
        elif tag == "title":
            self.match = True

    def handle_data(self, data: str) -> None:
        if self.match:
            self.metadata["title"] = data
            self.match = False

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata


"""OracleDocReader class"""


class OracleDocReader:
    """Read a file"""

    @staticmethod
    def generate_object_id(input_string: Union[str, None] = None) -> str:
        out_length = 32  # output length
        hash_len = 8  # hash value length

        if input_string is None:
            input_string = "".join(
                random.choices(
                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    k=16,
                )
            )

        # timestamp
        timestamp = int(time.time())
        timestamp_bin = struct.pack(">I", timestamp)  # 4 bytes

        # hash_value
        hashval_bin = hashlib.sha256(input_string.encode()).digest()
        hashval_bin = hashval_bin[:hash_len]  # 8 bytes

        # counter
        counter_bin = struct.pack(">I", random.getrandbits(32))  # 4 bytes

        # binary object id
        object_id = timestamp_bin + hashval_bin + counter_bin  # 16 bytes
        object_id_hex = object_id.hex()  # 32 bytes
        object_id_hex = object_id_hex.zfill(
            out_length
        )  # fill with zeros if less than 32 bytes

        object_id_hex = object_id_hex[:out_length]

        return object_id_hex

    @staticmethod
    def read_file(
        conn: Connection, file_path: str, params: dict
    ) -> Union[Document, None]:
        """Read a file using OracleReader
        Args:
            conn: Oracle Connection,
            file_path: Oracle Directory,
            params: ONNX file name.
        Returns:
            Plain text and metadata as Langchain Document.
        """

        metadata: Dict[str, Any] = {}
        try:
            import oracledb
        except ImportError as e:
            raise ImportError(
                "Unable to import oracledb, please install with "
                "`pip install -U oracledb`."
            ) from e
        try:
            oracledb.defaults.fetch_lobs = False
            cursor = conn.cursor()

            with open(file_path, "rb") as f:
                data = f.read()

            if data is None:
                return Document(page_content="", metadata=metadata)

            mdata = cursor.var(oracledb.DB_TYPE_CLOB)
            text = cursor.var(oracledb.DB_TYPE_CLOB)
            cursor.execute(
                """
                declare
                    input blob;
                begin
                    input := :blob;
                    :mdata := dbms_vector_chain.utl_to_text(input, json(:pref));
                    :text := dbms_vector_chain.utl_to_text(input);
                end;""",
                blob=data,
                pref=json.dumps(params),
                mdata=mdata,
                text=text,
            )
            cursor.close()

            if mdata is None:
                metadata = {}
            else:
                doc_data = str(mdata.getvalue())
                if doc_data.startswith("<!DOCTYPE html") or doc_data.startswith(
                    "<HTML>"
                ):
                    p = ParseOracleDocMetadata()
                    p.feed(doc_data)
                    metadata = p.get_metadata()

            doc_id = OracleDocReader.generate_object_id(conn.username + "$" + file_path)
            metadata["_oid"] = doc_id
            metadata["_file"] = file_path

            if text is None:
                return Document(page_content="", metadata=metadata)
            else:
                return Document(page_content=str(text.getvalue()), metadata=metadata)

        except Exception as ex:
            logger.info(f"An exception occurred :: {ex}")
            logger.info(f"Skip processing {file_path}")
            cursor.close()
            return None


"""OracleDocLoader class"""


class OracleDocLoader(BaseLoader):
    """Read documents using OracleDocLoader
    Args:
        conn: Oracle Connection,
        params: Loader parameters.
    """

    def __init__(self, conn: Connection, params: Dict[str, Any], **kwargs: Any):
        self.conn = conn
        self.params = json.loads(json.dumps(params))
        super().__init__(**kwargs)

    def load(self) -> List[Document]:
        """Load data into LangChain Document objects..."""
        try:
            import oracledb
        except ImportError as e:
            raise ImportError(
                "Unable to import oracledb, please install with "
                "`pip install -U oracledb`."
            ) from e

        ncols = 0
        results: List[Document] = []
        metadata: Dict[str, Any] = {}
        m_params = {"plaintext": "false"}
        try:
            # extract the parameters
            if self.params is not None:
                self.file = self.params.get("file")
                self.dir = self.params.get("dir")
                self.owner = self.params.get("owner")
                self.tablename = self.params.get("tablename")
                self.colname = self.params.get("colname")
            else:
                raise Exception("Missing loader parameters")

            oracledb.defaults.fetch_lobs = False

            if self.file:
                doc = OracleDocReader.read_file(self.conn, self.file, m_params)

                if doc is None:
                    return results

                results.append(doc)

            if self.dir:
                skip_count = 0
                for file_name in os.listdir(self.dir):
                    file_path = os.path.join(self.dir, file_name)
                    if os.path.isfile(file_path):
                        doc = OracleDocReader.read_file(self.conn, file_path, m_params)

                        if doc is None:
                            skip_count = skip_count + 1
                            logger.info(f"Total skipped: {skip_count}\n")
                        else:
                            results.append(doc)

            if self.tablename:
                try:
                    if self.owner is None or self.colname is None:
                        raise Exception("Missing owner or column name or both.")

                    cursor = self.conn.cursor()
                    self.mdata_cols = self.params.get("mdata_cols")
                    if self.mdata_cols is not None:
                        if len(self.mdata_cols) > 3:
                            raise Exception(
                                "Exceeds the max number of columns "
                                + "you can request for metadata."
                            )

                        # execute a query to get column data types
                        sql = (
                            "select column_name, data_type from all_tab_columns "
                            + "where owner = :ownername and "
                            + "table_name = :tablename"
                        )
                        cursor.execute(
                            sql,
                            ownername=self.owner.upper(),
                            tablename=self.tablename.upper(),
                        )

                        # cursor.execute(sql)
                        rows = cursor.fetchall()
                        for row in rows:
                            if row[0] in self.mdata_cols:
                                if row[1] not in [
                                    "NUMBER",
                                    "BINARY_DOUBLE",
                                    "BINARY_FLOAT",
                                    "LONG",
                                    "DATE",
                                    "TIMESTAMP",
                                    "VARCHAR2",
                                ]:
                                    raise Exception(
                                        "The datatype for the column requested "
                                        + "for metadata is not supported."
                                    )

                    self.mdata_cols_sql = ", rowid"
                    if self.mdata_cols is not None:
                        for col in self.mdata_cols:
                            self.mdata_cols_sql = self.mdata_cols_sql + ", " + col

                    # [TODO] use bind variables
                    sql = (
                        "select dbms_vector_chain.utl_to_text(t."
                        + self.colname
                        + ", json('"
                        + json.dumps(m_params)
                        + "')) mdata, dbms_vector_chain.utl_to_text(t."
                        + self.colname
                        + ") text"
                        + self.mdata_cols_sql
                        + " from "
                        + self.owner
                        + "."
                        + self.tablename
                        + " t"
                    )

                    cursor.execute(sql)
                    for row in cursor:
                        metadata = {}

                        if row is None:
                            doc_id = OracleDocReader.generate_object_id(
                                self.conn.username
                                + "$"
                                + self.owner
                                + "$"
                                + self.tablename
                                + "$"
                                + self.colname
                            )
                            metadata["_oid"] = doc_id
                            results.append(Document(page_content="", metadata=metadata))
                        else:
                            if row[0] is not None:
                                data = str(row[0])
                                if data.startswith("<!DOCTYPE html") or data.startswith(
                                    "<HTML>"
                                ):
                                    p = ParseOracleDocMetadata()
                                    p.feed(data)
                                    metadata = p.get_metadata()

                            doc_id = OracleDocReader.generate_object_id(
                                self.conn.username
                                + "$"
                                + self.owner
                                + "$"
                                + self.tablename
                                + "$"
                                + self.colname
                                + "$"
                                + str(row[2])
                            )
                            metadata["_oid"] = doc_id
                            metadata["_rowid"] = row[2]

                            # process projected metadata cols
                            if self.mdata_cols is not None:
                                ncols = len(self.mdata_cols)

                            for i in range(0, ncols):
                                metadata[self.mdata_cols[i]] = row[i + 2]

                            if row[1] is None:
                                results.append(
                                    Document(page_content="", metadata=metadata)
                                )
                            else:
                                results.append(
                                    Document(
                                        page_content=str(row[1]), metadata=metadata
                                    )
                                )
                except Exception as ex:
                    logger.info(f"An exception occurred :: {ex}")
                    traceback.print_exc()
                    cursor.close()
                    raise

            return results
        except Exception as ex:
            logger.info(f"An exception occurred :: {ex}")
            traceback.print_exc()
            raise


class OracleTextSplitter(TextSplitter):
    """Splitting text using Oracle chunker."""

    def __init__(self, conn: Connection, params: Dict[str, Any], **kwargs: Any) -> None:
        """Initialize."""
        self.conn = conn
        self.params = params
        super().__init__(**kwargs)
        try:
            import json

            try:
                import oracledb
            except ImportError as e:
                raise ImportError(
                    "Unable to import oracledb, please install with "
                    "`pip install -U oracledb`."
                ) from e

            self._oracledb = oracledb
            self._json = json
        except ImportError:
            raise ImportError(
                "oracledb or json or both are not installed. "
                + "Please install them. "
                + "Recommendations: `pip install oracledb`. "
            )

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""

        try:
            import oracledb
        except ImportError as e:
            raise ImportError(
                "Unable to import oracledb, please install with "
                "`pip install -U oracledb`."
            ) from e

        splits = []

        try:
            # returns strings or bytes instead of a locator
            self._oracledb.defaults.fetch_lobs = False

            cursor = self.conn.cursor()

            cursor.setinputsizes(content=oracledb.CLOB)
            cursor.execute(
                "select t.column_value from "
                + "dbms_vector_chain.utl_to_chunks(:content, json(:params)) t",
                content=text,
                params=self._json.dumps(self.params),
            )

            while True:
                row = cursor.fetchone()
                if row is None:
                    break
                d = self._json.loads(row[0])
                splits.append(d["chunk_data"])

            return splits

        except Exception as ex:
            logger.info(f"An exception occurred :: {ex}")
            traceback.print_exc()
            raise
