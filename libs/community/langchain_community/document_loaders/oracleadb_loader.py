from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class OracleAutonomousDatabaseLoader(BaseLoader):
    """
    Load from oracle adb

    Autonomous Database connection can be made by either connection_string
    or tns name. wallet_location and wallet_password are required
    for TLS connection.
    Each document will represent one row of the query result.
    Columns are written into the `page_content` and 'metadata' in
    constructor is written into 'metadata' of document,
    by default, the 'metadata' is None.
    """

    def __init__(
        self,
        query: str,
        user: str,
        password: str,
        *,
        schema: Optional[str] = None,
        tns_name: Optional[str] = None,
        config_dir: Optional[str] = None,
        wallet_location: Optional[str] = None,
        wallet_password: Optional[str] = None,
        connection_string: Optional[str] = None,
        metadata: Optional[List[str]] = None,
    ):
        """
        init method
        :param query: sql query to execute
        :param user: username
        :param password: user password
        :param schema: schema to run in database
        :param tns_name: tns name in tnsname.ora
        :param config_dir: directory of config files(tnsname.ora, wallet)
        :param wallet_location: location of wallet
        :param wallet_password: password of wallet
        :param connection_string: connection string to connect to adb instance
        :param metadata: metadata used in document
        """
        # Mandatory required arguments.
        self.query = query
        self.user = user
        self.password = password

        # Schema
        self.schema = schema

        # TNS connection Method
        self.tns_name = tns_name
        self.config_dir = config_dir

        # Wallet configuration is required for mTLS connection
        self.wallet_location = wallet_location
        self.wallet_password = wallet_password

        # Connection String connection method
        self.connection_string = connection_string

        # metadata column
        self.metadata = metadata

        # dsn
        self.dsn: Optional[str]
        self._set_dsn()

    def _set_dsn(self) -> None:
        if self.connection_string:
            self.dsn = self.connection_string
        elif self.tns_name:
            self.dsn = self.tns_name

    def _run_query(self) -> List[Dict[str, Any]]:
        try:
            import oracledb
        except ImportError as e:
            raise ImportError(
                "Could not import oracledb, "
                "please install with 'pip install oracledb'"
            ) from e
        connect_param = {"user": self.user, "password": self.password, "dsn": self.dsn}
        if self.dsn == self.tns_name:
            connect_param["config_dir"] = self.config_dir
        if self.wallet_location and self.wallet_password:
            connect_param["wallet_location"] = self.wallet_location
            connect_param["wallet_password"] = self.wallet_password

        try:
            connection = oracledb.connect(**connect_param)
            cursor = connection.cursor()
            if self.schema:
                cursor.execute(f"alter session set current_schema={self.schema}")
            cursor.execute(self.query)
            columns = [col[0] for col in cursor.description]
            data = cursor.fetchall()
            data = [
                {
                    i: (j if not isinstance(j, oracledb.LOB) else j.read())
                    for i, j in zip(columns, row)
                }
                for row in data
            ]
        except oracledb.DatabaseError as e:
            print("Got error while connecting: " + str(e))  # noqa: T201
            data = []
        finally:
            cursor.close()
            connection.close()

        return data

    def load(self) -> List[Document]:
        data = self._run_query()
        documents = []
        metadata_columns = self.metadata if self.metadata else []
        for row in data:
            metadata = {
                key: value for key, value in row.items() if key in metadata_columns
            }
            doc = Document(page_content=str(row), metadata=metadata)
            documents.append(doc)

        return documents
