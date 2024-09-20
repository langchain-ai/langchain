from types import TracebackType
from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
    get_pydantic_field_names,
)
from langchain_core.utils.env import env_var_is_set
from langchain_core.utils.utils import build_extra_kwargs
from pydantic import Field, SecretStr, model_validator
from snowflake.core import Root
from snowflake.snowpark import Session


class CortexSearchRetrieverError(Exception):
    """Error with the CortexSearchRetriever."""


class CortexSearchRetriever(BaseRetriever):
    """Snowflake Cortex Search Service document retriever.

    Setup:
        Install ``langchain-snowflake`` and set the following environment variables:
        - ``SNOWFLAKE_USERNAME``
        - ``SNOWFLAKE_PASSWORD`` (optionally, if not using "externalbrowser" authenticator)
        - ``SNOWFLAKE_ACCOUNT``
        - ``SNOWFLAKE_DATABASE``
        - ``SNOWFLAKE_SCHEMA``
        - ``SNOWFLAKE_ROLE``

        For example:

        .. code-block:: bash

            pip install -U langchain-snowflake
            export SNOWFLAKE_USERNAME="your-username"
            export SNOWFLAKE_PASSWORD="your-password"
            export SNOWFLAKE_ACCOUNT="your-account"
            export SNOWFLAKE_DATABASE="your-database"
            export SNOWFLAKE_SCHEMA="your-schema"
            export SNOWFLAKE_ROLE="your-role"


    Key init args:
        authenticator: str
            Authenticator method to utilize when logging into Snowflake. Refer to Snowflake documentation for more information.
        columns: List[str]
            List of columns to return in the search results.
        search_column: str
            Name of the search column in the Cortex Search Service.
        cortex_search_service: str
            Cortex search service to query against.
        filter: Dict[str, Any]
            Filter to apply to the search query.
        limit: int
            The maximum number of results to return in a single query.
        snowflake_username: str
            Snowflake username.
        snowflake_password: SecretStr
            Snowflake password.
        snowflake_account: str
            Snowflake account.
        snowflake_database: str
            Snowflake database.
        snowflake_schema: str
            Snowflake schema.
        snowflake_role: str
            Snowflake role.


    Instantiate:
        .. code-block:: python

            from langchain-snowflake import SnowflakeRetriever

            retriever = SnowflakeRetriever(
                authenticator="externalbrowser",
                columns=["name", "description", "era"],
                search_column="description",
                filter={"@eq": {"era": "Jurassic"}},
                search_service="dinosaur_svc",
            )

    Usage:
        .. code-block:: python

            query = "sharp teeth and claws"

            retriever.invoke(query)

    Use within a chain:
        .. code-block:: python

            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            from langchain_openai import ChatOpenAI

            prompt = ChatPromptTemplate.from_template(
                \"\"\"Answer the question based only on the context provided.

            Context: {context}

            Question: {question}\"\"\"
            )

            llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

            def format_docs(docs):
                return "\\n\\n".join(doc.page_content for doc in docs)

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            chain.invoke("Which dinosaur from the Jurassic period had sharp teeth and claws?")

    """  # noqa: E501

    sp_session: Optional[Session] = Field(alias="sp_session")
    """Snowpark session object."""

    _sp_root: Root
    """Snowpark API Root object."""

    search_column: str = Field()
    """Name of the search column in the Cortex Search Service. Always returned in the
    search results."""

    columns: List[str] = Field(default=[])
    """List of additional columns to return in the search results."""

    cortex_search_service: str = Field(alias="search_service")
    """Cortex search service to query against."""

    filter: Optional[Dict[str, Any]] = Field(default=None)
    """Filter to apply to the search query."""

    limit: Optional[int] = Field(default=None)
    """The maximum number of results to return in a single query."""

    snowflake_authenticator: Optional[str] = Field(default=None, alias="authenticator")
    """Authenticator method to utilize when logging into Snowflake. Refer to Snowflake
    documentation for more information."""

    snowflake_username: Optional[str] = Field(default=None, alias="username")
    """Automatically inferred from env var `SNOWFLAKE_USERNAME` if not provided."""

    snowflake_password: Optional[SecretStr] = Field(default=None, alias="password")
    """Automatically inferred from env var `SNOWFLAKE_PASSWORD` if not provided."""

    snowflake_account: Optional[str] = Field(default=None, alias="account")
    """Automatically inferred from env var `SNOWFLAKE_ACCOUNT` if not provided."""

    snowflake_database: Optional[str] = Field(default=None, alias="database")
    """Automatically inferred from env var `SNOWFLAKE_DATABASE` if not provided."""

    snowflake_schema: Optional[str] = Field(default=None, alias="schema")
    """Automatically inferred from env var `SNOWFLAKE_SCHEMA` if not provided."""

    snowflake_role: Optional[str] = Field(default=None, alias="role")
    """Automatically inferred from env var `SNOWFLAKE_ROLE` if not provided."""

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        values["model_kwargs"] = build_extra_kwargs(
            extra, values, all_required_field_names
        )
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate the environment needed to establish a Snowflake session or obtain
        an API root from a provided Snowflake session."""

        if "sp_session" not in values:
            values["username"] = get_from_dict_or_env(
                values, "username", "SNOWFLAKE_USERNAME"
            )
            values["account"] = get_from_dict_or_env(
                values, "account", "SNOWFLAKE_ACCOUNT"
            )
            values["database"] = get_from_dict_or_env(
                values, "database", "SNOWFLAKE_DATABASE"
            )
            values["schema"] = get_from_dict_or_env(
                values, "schema", "SNOWFLAKE_SCHEMA"
            )
            values["role"] = get_from_dict_or_env(values, "role", "SNOWFLAKE_ROLE")

            # check whether to authenticate with password or authenticator
            if "password" in values or env_var_is_set("SNOWFLAKE_PASSWORD"):
                values["password"] = convert_to_secret_str(
                    get_from_dict_or_env(values, "password", "SNOWFLAKE_PASSWORD")
                )
            elif "authenticator" in values or env_var_is_set("SNOWFLAKE_AUTHENTICATOR"):
                values["authenticator"] = get_from_dict_or_env(
                    values, "authenticator", "AUTHENTICATOR"
                )
                if values["authenticator"].lower() != "externalbrowser":
                    raise CortexSearchRetrieverError(
                        "Unable to authenticate. Unsupported authentication method"
                    )
            else:
                raise CortexSearchRetrieverError(
                    """Unable to authenticate. Please input Snowflake password directly
                    or as environment variables, or authenticate with via another
                    method by passing a valid `authenticator` type."""
                )

            connection_params = {
                "account": values["account"],
                "user": values["username"],
                "database": values["database"],
                "schema": values["schema"],
                "role": values["role"],
            }

            if "password" in values:
                connection_params["password"] = values["password"].get_secret_value()

            if "authenticator" in values:
                connection_params["authenticator"] = values["authenticator"]

            try:
                session = Session.builder.configs(connection_params).create()
                values["sp_session"] = session
            except Exception as e:
                raise CortexSearchRetrieverError(f"Failed to create session: {e}")

        return values

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._sp_root = Root(self.sp_session)

    def _columns(self, cols: List[str] = []) -> List[str]:
        """The columns to return in the search results."""
        override_cols = cols if cols else self.columns
        return [self.search_column] + override_cols

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Default query parameters for the Cortex Search Service retriever. Can be
        optionally overridden on each invocation of `invoke()`."""
        params: Dict[str, Any] = {}
        if self.filter:
            params["filter"] = self.filter
        if self.limit:
            params["limit"] = self.limit
        return params

    def _optional_params(self, **kwargs: Any) -> Dict[str, Any]:
        params = self._default_params
        params.update({k: v for k, v in kwargs.items() if k in ["filter", "limit"]})
        return params

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        try:
            response = (
                self._sp_root.databases[self.snowflake_database]
                .schemas[self.snowflake_schema]
                .cortex_search_services[self.cortex_search_service]
                .search(
                    query=str(query),
                    columns=self._columns(kwargs.get("columns", None)),
                    **self._optional_params(**kwargs),
                )
            )
            document_list = []
            for result in response.results:
                if self.search_column not in result.keys():
                    raise CortexSearchRetrieverError(
                        "Search column not found in Cortex Search response"
                    )
                else:
                    document_list.append(
                        self._create_document(result, self.search_column)
                    )
        except Exception as e:
            raise CortexSearchRetrieverError(f"Failed in search: {e}")

        return document_list

    def _create_document(self, response: Dict, search_column: str) -> Document:
        content = response.pop(search_column)
        doc = Document(page_content=content, metadata=response)

        return doc

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        try:
            if self.sp_session is not None:
                self.sp_session.close()
        except Exception as e:
            raise CortexSearchRetrieverError(f"Error while closing session: {e}")
        return None

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        raise NotImplementedError("Async is not supported for Snowflake Cortex Search")
