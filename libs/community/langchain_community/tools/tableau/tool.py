from typing import Optional

from langchain.prompts import PromptTemplate
from langchain_core.tools import tool, ToolException
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
# community imports from utilities and tools
from langchain_community.tools.tableau.prompt import vds_prompt, vds_response
from langchain_community.utilities.tableau import (
    env_vars_datasource_qa, 
    jwt_connected_app, 
    augment_datasource_metadata,
    get_headlessbi_data, 
    prepare_prompt_inputs
)


def initialize_datasource_qa(
    domain: Optional[str] = None,
    site: Optional[str] = None,
    jwt_client_id: Optional[str] = None,
    jwt_secret_id: Optional[str] = None,
    jwt_secret: Optional[str] = None,
    tableau_api_version: Optional[str] = None,
    tableau_user: Optional[str] = None,
    datasource_luid: Optional[str] = None,
    tooling_llm_model: Optional[str] = None
):
    """
    Initializes the Langgraph tool called 'datasource_qa' for analytical
    questions and answers on a Tableau Data Source

    Args:
        domain (Optional[str]): The domain of the Tableau server.
        site (Optional[str]): The site name on the Tableau server.
        jwt_client_id (Optional[str]): The client ID for JWT authentication.
        jwt_secret_id (Optional[str]): The secret ID for JWT authentication.
        jwt_secret (Optional[str]): The secret for JWT authentication.
        tableau_api_version (Optional[str]): The version of the Tableau API to use.
        tableau_user (Optional[str]): The Tableau user to authenticate as.
        datasource_luid (Optional[str]): The LUID of the data source to perform QA on.
        tooling_llm_model (Optional[str]): The LLM model to use for tooling operations.

    Returns:
        function: A decorated function that can be used as a langgraph tool for data source QA.

    The returned function (datasource_qa) takes the following parameters:
        user_input (str): The user's query or command represented in simple SQL.
        previous_call_error (Optional[str]): Any error from a previous call, for error handling.

    It returns a dictionary containing the results of the QA operation.

    Note:
        If arguments are not provided, the function will attempt to read them from
        environment variables, typically stored in a .env file.
    """
    try:
        from langchain_openai import ChatOpenAI
        from pydantic import BaseModel, Field
        # if arguments are not provided, the tool obtains environment variables directly from .env
        env_vars = env_vars_datasource_qa(
            domain=domain,
            site=site,
            jwt_client_id=jwt_client_id,
            jwt_secret_id=jwt_secret_id,
            jwt_secret=jwt_secret,
            tableau_api_version=tableau_api_version,
            tableau_user=tableau_user,
            datasource_luid=datasource_luid,
            tooling_llm_model=tooling_llm_model
        )

        class SimpleDataSourceQAInputs(BaseModel):
            """Describes inputs for usage of the datasource_qa tool"""

            user_input: str = Field(
                ...,
                description="The user question, query, command or instruction that can be answered or executed with data accessed by this tool",
                examples=[
                    "I would like to know the average discount, total sales, number of orders and profits by region sorted by profit so I can evaluate regional performance"
                ]
            )
            previous_call_error: Optional[str] = Field(
                None,
                description="If the previous tool call resulted in a VizQL Data Service error suggesting a malformed query, include the error when retrying this tool to diagnose the problem otherwise use None.",
                examples=[
                    None, # no errors example
                    "Error: Quantitative Filters must have a QuantitativeFilterType"
                ],
            )
            previous_call_query: Optional[str] = Field(
                None,
                description="If the previous tool call resulted in a VizQL Data Service error suggesting a malformed query, include the faulty query when retrying this tool to diagnose the problem otherwise use None.",
                examples=[
                    None, # no errors example
                    "{\"fields\":[{\"fieldCaption\":\"Sub-Category\",\"fieldAlias\":\"SubCategory\",\"sortDirection\":\"DESC\",\"sortPriority\":1},{\"function\":\"SUM\",\"fieldCaption\":\"Sales\",\"fieldAlias\":\"TotalSales\"}],\"filters\":[{\"field\":{\"fieldCaption\":\"Order Date\"},\"filterType\":\"QUANTITATIVE_DATE\",\"minDate\":\"2023-04-01\",\"maxDate\":\"2023-10-01\"},{\"field\":{\"fieldCaption\":\"Sales\"},\"filterType\":\"QUANTITATIVE_NUMERICAL\",\"quantitativeFilterType\":\"MIN\",\"min\":200000},{\"field\":{\"fieldCaption\":\"Sub-Category\"},\"filterType\":\"MATCH\",\"exclude\":true,\"contains\":\"Technology\"}]}"
                ],
            )


        @tool("datasource_qa", args_schema=SimpleDataSourceQAInputs)
        def datasource_qa(
            user_input: str,
            previous_call_error: Optional[str] = None,
            previous_call_query: Optional[str] = None
        ) -> dict:
            """
            Queries a Tableau data source for analytical Q&A. Returns a data set you can use to answer user questions.
            You need a data source to target to use this tool. If a target data source is unknown, use a data source
            search tool to find the right resource and retry with more information or ask the user to provide it.

            Prioritize this tool if the user asks you to analyze and explore data. This tool includes Agent summarization
            and is not meant for direct data set exports. To be more efficient, query all the data you need in a single
            request rather than selecting small slices of data in multiple requests.

            If you received an error after using this tool, mention it in your next attempt to help the tool correct itself.
            """

            # Session scopes are limited to only required authorizations to Tableau resources that support tool operations
            access_scopes = [
                "tableau:content:read", # for quering Tableau Metadata API
                "tableau:viz_data_service:read" # for querying VizQL Data Service
            ]
            try:
                tableau_session = jwt_connected_app(
                    tableau_domain=env_vars["domain"],
                    tableau_site=env_vars["site"],
                    jwt_client_id=env_vars["jwt_client_id"],
                    jwt_secret_id=env_vars["jwt_secret_id"],
                    jwt_secret=env_vars["jwt_secret"],
                    tableau_api=env_vars["tableau_api_version"],
                    tableau_user=env_vars["tableau_user"],
                    scopes=access_scopes
                )
            except Exception as e:
                auth_error_string = f"""
                CRITICAL ERROR: Could not authenticate to the Tableau site successfully.
                This tool is unusable as a result.
                Error from remote server: {e}

                INSTRUCTION: Do not ask the user to provide credentials directly or in chat since they should
                originate from a secure Connected App or similar authentication mechanism. You may inform the
                user that you are not able to access their Tableau environment at this time. You can also describe
                the nature of the error to help them understand why you can't service their request.
                """
                raise ToolException(auth_error_string)

            # credentials to access Tableau environment on behalf of the user
            tableau_auth =  tableau_session['credentials']['token']

            # Data source for VDS querying
            tableau_datasource = env_vars["datasource_luid"]

            # 1. Initialize Langchain chat template with an augmented prompt containing metadata for the datasource
            query_data_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content = augment_datasource_metadata(
                    api_key = tableau_auth,
                    url = domain,
                    datasource_luid = tableau_datasource,
                    prompt = vds_prompt,
                    previous_errors = previous_call_error,
                    previous_error_query = previous_call_query
                )),
                ("user", "{utterance}")
            ])

            # 2. Instantiate language model to execute the prompt to write a VizQL Data Service query
            query_writer = ChatOpenAI(
                model=env_vars["tooling_llm_model"],
                temperature=0
            )

            # 3. Query data from Tableau's VizQL Data Service using the AI written payload
            def get_data(vds_query):
                try:
                    data = get_headlessbi_data(
                        api_key = tableau_auth,
                        url = domain,
                        datasource_luid = tableau_datasource,
                        payload = vds_query.content
                    )

                    return {
                        "vds_query": vds_query,
                        "data_table": data,
                    }
                except Exception as e:
                    query_error_message = f"""
                    Tableau's VizQL Data Service return an error for the generated query:
                    {str(vds_query.content)}

                    The user_input used to write this query was:
                    {str(user_input)}

                    This was the error:
                    {str(e)}

                    Consider retrying this tool with the same `user_input` key but include the query and
                    the error in the `previous_call_error` key for the tool to debug the query.
                    """

                    raise ToolException(query_error_message)

            # 4. Prepare inputs for a structured response to the calling Agent
            def response_inputs(input):
                data = {
                    "query": input.get('vds_query', ''),
                    "data_source": tableau_datasource,
                    "data_table": input.get('data_table', ''),
                }
                inputs = prepare_prompt_inputs(data=data, user_string=user_input)
                return inputs

            # 5. Response template for the Agent with further instructions
            enhanced_prompt = PromptTemplate(
                input_variables=["data_source", "vds_query", "data_table", "user_input"],
                template=vds_response
            )

            # this chain defines the flow of data through the system
            chain = query_data_prompt | query_writer | get_data | response_inputs | enhanced_prompt

            # invoke the chain to generate a query and obtain data
            vizql_data = chain.invoke(user_input)

            # Return the structured output
            return vizql_data

        return datasource_qa
    
    except ImportError as e:
        # Provide a *very clear* message about *all* missing dependencies
        missing_deps = []
        if "langchain_openai" in str(e): missing_deps.append("langchain-openai")
        if "pydantic" in str(e): missing_deps.append("pydantic")

        raise ImportError(
            f"Missing dependencies: {', '.join(missing_deps)}. Please install them using `pip install {' '.join(missing_deps)}` or similar."
        )
    except Exception as e: # Catch potential API errors as well
        raise ValueError(f"Error in my_tool: {e}")
