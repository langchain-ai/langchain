from typing import Any, Dict, List, Optional

from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
)
from langchain_core.utils.env import env_var_is_set


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

    _sp_session: Any = None
    """Snowpark session object."""

    _sp_root: Any = None
    """Snowpark API Root object."""

    authenticator: Optional[str] = Field(default=None, alias="authenticator")
    """Authenticator method to utilize when logging into Snowflake. Refer to Snowflake documentation for more information."""

    search_column: str = Field(default=None, alias="search_column")
    """Name of the search column in the Cortex Search Service. Always returned in the search results."""

    columns: Optional[List[str]] = Field(default=[], alias="columns")
    """List of additional columns to return in the search results."""

    cortex_search_service: str = Field(default=None, alias="search_service")
    """Cortex search service to query against."""

    filter: Optional[Dict[str, Any]] = None
    """Filter to apply to the search query."""

    limit: Optional[int] = None
    """The maximum number of results to return in a single query."""

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
        if values["snowflake_password"] is not None or env_var_is_set(
            "SNOWFLAKE_PASSWORD"
        ):
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
                """Unable to authenticate. Please input Snowflake password directly as env variable, or authenticate with externalbrowser."""
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
        values["snowflake_role"] = get_from_dict_or_env(
            values, "snowflake_role", "SNOWFLAKE_ROLE"
        )

        connection_params = {
            "account": values["snowflake_account"],
            "user": values["snowflake_username"],
            "password": values["snowflake_password"].get_secret_value(),
            "database": values["snowflake_database"],
            "schema": values["snowflake_schema"],
            "role": values["snowflake_role"],
        }

        # specify connection params based on if password/authenticator is provided
        if values["snowflake_password"] is not None:
            connection_params["password"] = values[
                "snowflake_password"
            ].get_secret_value()
        else:
            connection_params["authenticator"] = values["authenticator"]

        # Check other required fields
        if values["search_column"] is None:
            raise CortexSearchRetrieverError("Search column not provided")
        if values["cortex_search_service"] is None:
            raise CortexSearchRetrieverError("Cortex search service not provided")

        # Attempt to create a session
        try:
            values["_sp_session"] = Session.builder.configs(connection_params).create()
        except Exception as e:
            raise CortexSearchRetrieverError(f"Failed to create session: {e}")

        try:
            values["_sp_root"] = Root(values["_sp_session"])
        except Exception as e:
            raise CortexSearchRetrieverError(f"Failed to initialize Root: {e}")

        return values

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        try:
            kwargs = {"columns": self.columns if self.columns else [self.search_column]}

            if self.filter:
                kwargs["filter"] = self.filter
            if self.limit:
                kwargs["limit"] = self.limit

            responses = (
                self._sp_root.databases[self.snowflake_database]
                .schemas[self.snowflake_schema]
                .cortex_search_services[self.cortex_search_service]
                .search(
                    query,
                    **kwargs,
                )
            )

            document_list = []
            for response in responses.results:
                if self.search_column not in response.keys():
                    raise CortexSearchRetrieverError(
                        "Search column not found in Cortex Search response"
                    )
                else:
                    document_list.append(
                        self._create_document(response, self.search_column)
                    )
        except Exception as e:
            raise CortexSearchRetrieverError(f"Failed in search: {e}")

        return document_list

    def _create_document(self, response: Dict, search_column: str) -> Document:
        content = response.pop(search_column)
        doc = Document(page_content=content, metadata=response)

        return doc

    def __del__(self) -> None:
        if getattr(self, "_sp_session", None) is not None:
            self._sp_session.close()

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        raise NotImplementedError("error")
