from typing import Any, Dict, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
    get_pydantic_field_names,
)
from langchain_core.utils.env import env_var_is_set
from langchain_core.utils.utils import build_extra_kwargs

# questions:
# 1. is the filter argument to Cortex Search always of the format Dict[str, Dict[str, str]]?


class CortexSearchRetrieverError(Exception):
    """Error with Snowpark client."""


class CortexSearchRetriever(BaseRetriever):
    """Retriever for Snowflake Cortex Search."""

    _sp_session: Any = None
    """Snowpark session object."""

    _sp_root: Any = None
    """Snowpark API Root object."""

    authenticator: Optional[str] = Field(default=None, alias="authenticator")
    """Authenticator to utilize when logging into Snowflake.
        Refer to docs for more options."""

    columns: Optional[List[Any]] = Field(default=None, alias="columns")
    """Columns to search when using the Search Service.
        Refer to docs for more options."""

    cortex_search_service: Optional[str] = Field(default=None, alias="search_service")
    """Cortex search service to use, stored in Snowflake database.
        Refer to docs for more options."""

    filter: Optional[Dict[str, Any]] = Field(default=None, alias="filter")
    """Filter to use when querying the Cortex Search Service.
        Refer to docs for more options."""

    limit: Optional[int] = Field(default=None, alias="limit")
    """Limit of number of responses to obtain when querying the Cortex Search Service.
        Refer to docs for more options."""

    snowflake_username: Optional[str] = Field(default=None, alias="username")
    """Automatically inferred from env var `SNOWFLAKE_USERNAME` if not provided."""

    snowflake_password: Optional[SecretStr] = Field(default=None, alias="password")
    """Automatically inferred from env var `SNOWFLAKE_PASSWORD` if not provided."""

    snowflake_account: Optional[str] = Field(default=None, alias="account")
    """Automatically inferred from env var `SNOWFLAKE_ACCOUNT` if not provided."""

    snowflake_database: Optional[str] = Field(default=None, alias="database")
    """Database in which Cortex Search Service is stored.
    Automatically inferred from env var `SNOWFLAKE_DATABASE` if not provided."""

    snowflake_schema: Optional[str] = Field(default=None, alias="schema")
    """Schema in which Cortex Search Service is stored.
    Automatically inferred from env var `SNOWFLAKE_SCHEMA` if not provided."""

    snowflake_warehouse: Optional[str] = Field(default=None, alias="warehouse")
    """Automatically inferred from env var `SNOWFLAKE_WAREHOUSE` if not provided."""

    snowflake_role: Optional[str] = Field(default=None, alias="role")
    """Automatically inferred from env var `SNOWFLAKE_ROLE` if not provided."""

    @root_validator(pre=True)
    def build_extra(cls, values: Dict) -> Dict:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        values["model_kwargs"] = build_extra_kwargs(
            extra, values, all_required_field_names
        )
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        try:
            from snowflake.core import Root
            from snowflake.snowpark import Session
        except ImportError:
            raise ImportError(
                "`snowflake-snowpark-python` package not found, please install it with "
                "`pip install snowflake-snowpark-python`"
            )

        values["snowflake_username"] = get_from_dict_or_env(
            values, "snowflake_username", "SNOWFLAKE_USERNAME"
        )
        # check whether to authenticate with password or authenticator
        if values["snowflake_password"] is not None or env_var_is_set("SNOWFLAKE_PASSWORD"):
            values["snowflake_password"] = convert_to_secret_str(
                get_from_dict_or_env(values, "snowflake_password", "SNOWFLAKE_PASSWORD")
            )
        elif values["authenticator"] is not None:
            values["authenticator"] = get_from_dict_or_env(
                values, "authenticator", "AUTHENTICATOR"
            )
            if values["authenticator"].lower() != "externalbrowser":
                raise CortexSearchRetrieverError(
                    "Unable to authenticate. Unsupported authentication method"
                )
            # check if authentication method is supported
        else:
            raise CortexSearchRetrieverError(
                "Unable to authenticate. Please input Snowflake password directly/as env variable, or authenticate with externalbrowser."
            )
        values["snowflake_account"] = get_from_dict_or_env(
            values, "snowflake_account", "SNOWFLAKE_ACCOUNT"
        )
        values["snowflake_database"] = get_from_dict_or_env(
            values, "snowflake_database", "SNOWFLAKE_DATABASE"
        )
        values["snowflake_schema"] = get_from_dict_or_env(
            values, "snowflake_schema", "SNOWFLAKE_SCHEMA"
        )
        values["snowflake_warehouse"] = get_from_dict_or_env(
            values, "snowflake_warehouse", "SNOWFLAKE_WAREHOUSE"
        )
        values["snowflake_role"] = get_from_dict_or_env(
            values, "snowflake_role", "SNOWFLAKE_ROLE"
        )

        connection_params = {
            "account": values["snowflake_account"],
            "user": values["snowflake_username"],
            #    "password": values["snowflake_password"].get_secret_value(),
            "database": values["snowflake_database"],
            "schema": values["snowflake_schema"],
            "warehouse": values["snowflake_warehouse"],
            "role": values["snowflake_role"],
        }
        # specify connection params based on if password/authenticator is provided
        if values["snowflake_password"] is not None:
            connection_params["password"] = values[
                "snowflake_password"
            ].get_secret_value()
        else:
            connection_params["authenticator"] = values["authenticator"]

        try:
            values["_sp_session"] = Session.builder.configs(connection_params).create()
        except Exception as e:
            raise CortexSearchRetrieverError(f"Failed to create session: {e}")

        try:
            values["_sp_root"] = Root(values["_sp_session"])
        except Exception as e:
            raise CortexSearchRetrieverError(f"Failed to initialize Root: {e}")

        return values

    def _create_document(self, response: Dict, search_column: str) -> Document:
        content = response.pop(search_column)
        doc = Document(page_content=content, metadata=response)

        return doc

    def _get_relevant_documents(
        self,
        query: str,
        search_column: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        try:
            responses = (
                self._sp_root.databases[self.snowflake_database]
                .schemas[self.snowflake_schema]
                .cortex_search_services[self.cortex_search_service]
                .search(
                    query=query,
                    columns=self.columns,
                    filter=self.filter,
                    limit=self.limit,
                )
            )

            document_list = []
            for response in responses.results:
                if search_column not in response.keys():
                    raise CortexSearchRetrieverError(
                        "Search column not found in Cortex Search response"
                    )
                else:
                    document_list.append(self._create_document(response, search_column))
        except Exception as e:
            raise CortexSearchRetrieverError("Failed in search: {e}")

        return document_list

    def __del__(self) -> None:
        if getattr(self, "_sp_session", None) is not None:
            self._sp_session.close()

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        raise NotImplementedError("error")
