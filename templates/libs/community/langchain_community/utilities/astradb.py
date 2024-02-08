from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from astrapy.db import (
        AstraDB,
        AsyncAstraDB,
    )


class AstraDBEnvironment:
    def __init__(
        self,
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        astra_db_client: Optional[AstraDB] = None,
        async_astra_db_client: Optional[AsyncAstraDB] = None,
        namespace: Optional[str] = None,
    ) -> None:
        self.token = token
        self.api_endpoint = api_endpoint
        astra_db = astra_db_client
        self.async_astra_db = async_astra_db_client
        self.namespace = namespace

        from astrapy import db

        try:
            from astrapy.db import AstraDB
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Could not import a recent astrapy python package. "
                "Please install it with `pip install --upgrade astrapy`."
            )

        supports_async = hasattr(db, "AsyncAstraDB")

        # Conflicting-arg checks:
        if astra_db_client is not None or async_astra_db_client is not None:
            if token is not None or api_endpoint is not None:
                raise ValueError(
                    "You cannot pass 'astra_db_client' or 'async_astra_db_client' to "
                    "AstraDBEnvironment if passing 'token' and 'api_endpoint'."
                )

        if token and api_endpoint:
            astra_db = AstraDB(
                token=self.token,
                api_endpoint=self.api_endpoint,
                namespace=self.namespace,
            )
            if supports_async:
                self.async_astra_db = db.AsyncAstraDB(
                    token=self.token,
                    api_endpoint=self.api_endpoint,
                    namespace=self.namespace,
                )

        if astra_db:
            self.astra_db = astra_db
        else:
            if self.async_astra_db:
                self.astra_db = AstraDB(
                    token=self.async_astra_db.token,
                    api_endpoint=self.async_astra_db.base_url,
                    api_path=self.async_astra_db.api_path,
                    api_version=self.async_astra_db.api_version,
                    namespace=self.async_astra_db.namespace,
                )
            else:
                raise ValueError(
                    "Must provide 'astra_db_client' or 'async_astra_db_client' or "
                    "'token' and 'api_endpoint'"
                )

        if not self.async_astra_db and self.astra_db and supports_async:
            self.async_astra_db = db.AsyncAstraDB(
                token=self.astra_db.token,
                api_endpoint=self.astra_db.base_url,
                api_path=self.astra_db.api_path,
                api_version=self.astra_db.api_version,
                namespace=self.astra_db.namespace,
            )
