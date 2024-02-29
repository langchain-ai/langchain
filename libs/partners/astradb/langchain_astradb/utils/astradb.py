from __future__ import annotations

import asyncio
import inspect
import json
import warnings
from asyncio import InvalidStateError, Task
from enum import Enum
from typing import Any, Awaitable, Dict, List, Optional, Union

import langchain_core
from astrapy.api import APIRequestError
from astrapy.db import AstraDB, AsyncAstraDB


class SetupMode(Enum):
    SYNC = 1
    ASYNC = 2
    OFF = 3


class _AstraDBEnvironment:
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
        async_astra_db = async_astra_db_client
        self.namespace = namespace

        # Conflicting-arg checks:
        if astra_db_client is not None or async_astra_db_client is not None:
            if token is not None or api_endpoint is not None:
                raise ValueError(
                    "You cannot pass 'astra_db_client' or 'async_astra_db_client' to "
                    "AstraDBEnvironment if passing 'token' and 'api_endpoint'."
                )

        if token and api_endpoint:
            astra_db = AstraDB(
                token=token,
                api_endpoint=api_endpoint,
                namespace=self.namespace,
            )
            async_astra_db = AsyncAstraDB(
                token=token,
                api_endpoint=api_endpoint,
                namespace=self.namespace,
            )

        if astra_db:
            self.astra_db = astra_db.copy()
            if async_astra_db:
                self.async_astra_db = async_astra_db.copy()
            else:
                self.async_astra_db = self.astra_db.to_async()
        elif async_astra_db:
            self.async_astra_db = async_astra_db.copy()
            self.astra_db = self.async_astra_db.to_sync()
        else:
            raise ValueError(
                "Must provide 'astra_db_client' or 'async_astra_db_client' or "
                "'token' and 'api_endpoint'"
            )

        self.astra_db.set_caller(
            caller_name="langchain",
            caller_version=getattr(langchain_core, "__version__", None),
        )
        self.async_astra_db.set_caller(
            caller_name="langchain",
            caller_version=getattr(langchain_core, "__version__", None),
        )


class _AstraDBCollectionEnvironment(_AstraDBEnvironment):
    def __init__(
        self,
        collection_name: str,
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        astra_db_client: Optional[AstraDB] = None,
        async_astra_db_client: Optional[AsyncAstraDB] = None,
        namespace: Optional[str] = None,
        setup_mode: SetupMode = SetupMode.SYNC,
        pre_delete_collection: bool = False,
        embedding_dimension: Union[int, Awaitable[int], None] = None,
        metric: Optional[str] = None,
        requested_indexing_policy: Optional[Dict[str, Any]] = None,
        default_indexing_policy: Optional[Dict[str, Any]] = None,
    ) -> None:
        from astrapy.db import AstraDBCollection, AsyncAstraDBCollection

        super().__init__(
            token, api_endpoint, astra_db_client, async_astra_db_client, namespace
        )
        self.collection_name = collection_name
        self.collection = AstraDBCollection(
            collection_name=collection_name,
            astra_db=self.astra_db,
        )

        self.async_collection = AsyncAstraDBCollection(
            collection_name=collection_name,
            astra_db=self.async_astra_db,
        )

        if requested_indexing_policy is not None:
            _options = {"indexing": requested_indexing_policy}
        else:
            _options = None

        self.async_setup_db_task: Optional[Task] = None
        if setup_mode == SetupMode.ASYNC:
            async_astra_db = self.async_astra_db

            async def _setup_db() -> None:
                if pre_delete_collection:
                    await async_astra_db.delete_collection(collection_name)
                if inspect.isawaitable(embedding_dimension):
                    dimension = await embedding_dimension
                else:
                    dimension = embedding_dimension

                try:
                    await async_astra_db.create_collection(
                        collection_name,
                        dimension=dimension,
                        metric=metric,
                        options=_options,
                    )
                except (APIRequestError, ValueError):
                    # possibly the collection is preexisting and may have legacy,
                    # or custom, indexing settings: verify
                    get_coll_response = await async_astra_db.get_collections(
                        options={"explain": True}
                    )
                    collections = (get_coll_response["status"] or {}).get(
                        "collections"
                    ) or []
                    if not self._validate_indexing_policy(
                        detected_collections=collections,
                        collection_name=self.collection_name,
                        requested_indexing_policy=requested_indexing_policy,
                        default_indexing_policy=default_indexing_policy,
                    ):
                        # other reasons for the exception
                        raise

            self.async_setup_db_task = asyncio.create_task(_setup_db())
        elif setup_mode == SetupMode.SYNC:
            if pre_delete_collection:
                self.astra_db.delete_collection(collection_name)
            if inspect.isawaitable(embedding_dimension):
                raise ValueError(
                    "Cannot use an awaitable embedding_dimension with async_setup "
                    "set to False"
                )
            else:
                try:
                    self.astra_db.create_collection(
                        collection_name,
                        dimension=embedding_dimension,  # type: ignore[arg-type]
                        metric=metric,
                        options=_options,
                    )
                except (APIRequestError, ValueError):
                    # possibly the collection is preexisting and may have legacy,
                    # or custom, indexing settings: verify
                    get_coll_response = self.astra_db.get_collections(  # type: ignore[union-attr]
                        options={"explain": True}
                    )
                    collections = (get_coll_response["status"] or {}).get(
                        "collections"
                    ) or []
                    if not self._validate_indexing_policy(
                        detected_collections=collections,
                        collection_name=self.collection_name,
                        requested_indexing_policy=requested_indexing_policy,
                        default_indexing_policy=default_indexing_policy,
                    ):
                        # other reasons for the exception
                        raise

    @staticmethod
    def _validate_indexing_policy(
        detected_collections: List[Dict[str, Any]],
        collection_name: str,
        requested_indexing_policy: Optional[Dict[str, Any]],
        default_indexing_policy: Optional[Dict[str, Any]],
    ) -> bool:
        """
        This is a validation helper, to be called when the collection-creation
        call has failed.

        Args:
            detected_collection (List[Dict[str, Any]]):
                the list of collection items returned by astrapy
            collection_name (str): the name of the collection whose attempted
                creation failed
            requested_indexing_policy: the 'indexing' part of the collection
                options, e.g. `{"deny": ["field1", "field2"]}`.
                Leave to its default of None if no options required.
            default_indexing_policy: an optional 'default value' for the
                above, used to issue just a gentle warning in the special
                case that no policy is detected on a preexisting collection
                on DB and the default is requested. This is to enable
                a warning-only transition to new code using indexing without
                disrupting usage of a legacy collection, i.e. one created
                before adopting the usage of indexing policies altogether.
                You cannot pass this one without requested_indexing_policy.

        This function may raise an error (indexing mismatches), issue a warning
        (about legacy collections), or do nothing.
        In any case, when the function returns, it returns either
            - True: the exception was handled here as part of the indexing
              management
            - False: the exception is unrelated to indexing and the caller
              has to reraise it.
        """
        if requested_indexing_policy is None and default_indexing_policy is not None:
            raise ValueError(
                "Cannot specify a default indexing policy "
                "when no indexing policy is requested for this collection "
                "(requested_indexing_policy is None, "
                "default_indexing_policy is not None)."
            )

        preexisting = [
            collection
            for collection in detected_collections
            if collection["name"] == collection_name
        ]
        if preexisting:
            pre_collection = preexisting[0]
            # if it has no "indexing", it is a legacy collection
            pre_col_options = pre_collection.get("options") or {}
            if "indexing" not in pre_col_options:
                # legacy collection on DB
                if requested_indexing_policy == default_indexing_policy:
                    warnings.warn(
                        (
                            f"Astra DB collection '{collection_name}' is "
                            "detected as legacy and has indexing turned "
                            "on for all fields. This implies stricter "
                            "limitations on the amount of text each string in a "
                            "document can store. Consider reindexing anew on a "
                            "fresh collection to be able to store longer texts."
                        ),
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    raise ValueError(
                        f"Astra DB collection '{collection_name}' is "
                        "detected as legacy and has indexing turned "
                        "on for all fields. This is incompatible with "
                        "the requested indexing policy for this object. "
                        "Consider reindexing anew on a fresh "
                        "collection with the requested indexing "
                        "policy, or alternatively leave the indexing "
                        "settings for this object to their defaults "
                        "to keep using this collection."
                    )
            elif pre_col_options["indexing"] != requested_indexing_policy:
                # collection on DB has indexing settings, but different
                options_json = json.dumps(pre_col_options["indexing"])
                if pre_col_options["indexing"] == default_indexing_policy:
                    default_desc = " (default setting)"
                else:
                    default_desc = ""
                raise ValueError(
                    f"Astra DB collection '{collection_name}' is "
                    "detected as having the following indexing policy: "
                    f"{options_json}{default_desc}. This is incompatible "
                    "with the requested indexing policy for this object. "
                    "Consider reindexing anew on a fresh "
                    "collection with the requested indexing "
                    "policy, or alternatively align the requested "
                    "indexing settings to the collection to keep using it."
                )
            else:
                # the discrepancies have to do with options other than indexing
                return False
            # the original exception, related to indexing, was handled here
            return True
        else:
            # foreign-origin for the original exception
            return False

    def ensure_db_setup(self) -> None:
        if self.async_setup_db_task:
            try:
                self.async_setup_db_task.result()
            except InvalidStateError:
                raise ValueError(
                    "Asynchronous setup of the DB not finished. "
                    "NB: AstraDB components sync methods shouldn't be called from the "
                    "event loop. Consider using their async equivalents."
                )

    async def aensure_db_setup(self) -> None:
        if self.async_setup_db_task:
            await self.async_setup_db_task
