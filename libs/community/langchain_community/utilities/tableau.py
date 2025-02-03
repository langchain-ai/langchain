import os
import json
import re
import logging
from typing import Dict, Optional, Any, List
from dotenv import load_dotenv
import requests
import jwt
from datetime import datetime, timedelta, timezone
from uuid import uuid4


def jwt_connected_app(
        tableau_domain: str,
        tableau_site: str,
        tableau_api: str,
        tableau_user: str,
        jwt_client_id: str,
        jwt_secret_id: str,
        jwt_secret: str,
        scopes: List[str],
) -> Dict[str, Any]:
    """
    Authenticates a user to Tableau using JSON Web Token (JWT) authentication.

    This function generates a JWT based on the provided credentials and uses it to authenticate
    a user with the Tableau Server or Tableau Online. The JWT is created with a specified expiration
    time and scopes, allowing for secure access to Tableau resources.

    Args:
        tableau_domain (str): The domain of the Tableau Server or Tableau Online instance.
        tableau_site (str): The content URL of the specific Tableau site to authenticate against.
        tableau_api (str): The version of the Tableau API to use for authentication.
        tableau_user (str): The username of the Tableau user to authenticate.
        jwt_client_id (str): The client ID used for generating the JWT.
        jwt_secret_id (str): The key ID associated with the JWT secret.
        jwt_secret (str): The secret key used to sign the JWT.
        scopes (List[str]): A list of scopes that define the permissions granted by the JWT.

    Returns:
        Dict[str, Any]: A dictionary containing the response from the Tableau authentication endpoint,
        typically including an API key or session that is valid for 2 hours and user information.
    """
    # Encode the payload and secret key to generate the JWT
    token = jwt.encode(
        {
        "iss": jwt_client_id,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        "jti": str(uuid4()),
        "aud": "tableau",
        "sub": tableau_user,
        "scp": scopes
        },
        jwt_secret,
        algorithm = "HS256",
        headers = {
        'kid': jwt_secret_id,
        'iss': jwt_client_id
        }
    )

    # authentication endpoint + request headers & payload
    endpoint = f"{tableau_domain}/api/{tableau_api}/auth/signin"

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    payload = {
        "credentials": {
        "jwt": token,
        "site": {
            "contentUrl": tableau_site,
        }
        }
    }

    response = requests.post(endpoint, headers=headers, json=payload)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        return response.json()
    else:
        error_message = (
            f"Failed to authenticate to the Tableau site. "
            f"Status code: {response.status_code}. Response: {response.text}"
        )
        raise RuntimeError(error_message)


def get_headlessbi_data(payload: str, url: str, api_key: str, datasource_luid: str):
    json_payload = json.loads(payload)

    try:
        headlessbi_data = query_vds(
            api_key=api_key,
            datasource_luid=datasource_luid,
            url=url,
            query=json_payload
        )

        if not headlessbi_data or 'data' not in headlessbi_data:
            raise ValueError("Invalid or empty response from query_vds")

        markdown_table = json_to_markdown_table(headlessbi_data['data'])
        return markdown_table

    except ValueError as ve:
        logging.error(f"Value error in get_headlessbi_data: {str(ve)}")
        raise

    except json.JSONDecodeError as je:
        logging.error(f"JSON decoding error in get_headlessbi_data: {str(je)}")
        raise ValueError("Invalid JSON format in the payload")

    except Exception as e:
        logging.error(f"Unexpected error in get_headlessbi_data: {str(e)}")
        raise RuntimeError(f"An unexpected error occurred: {str(e)}")


def get_payload(output):
    try:
        parsed_output = output.split('JSON_payload')[1]
    except IndexError:
        raise ValueError("'JSON_payload' not found in the output")

    match = re.search(r'{.*}', parsed_output, re.DOTALL)
    if match:
        json_string = match.group(0)
        try:
            payload = json.loads(json_string)
            return payload
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in the payload")
    else:
        raise ValueError("No JSON payload found in the parsed output")


def get_values(api_key: str, url: str, datasource_luid: str, caption: str):
    column_values = {'fields': [{'fieldCaption': caption}]}
    output = query_vds(
        api_key=api_key,
        datasource_luid=datasource_luid,
        url=url,
        query=column_values
    )
    if output is None:
        return None
    sample_values = [list(item.values())[0] for item in output['data']][:4]
    return sample_values


def augment_datasource_metadata(
    api_key: str,
    url: str,
    datasource_luid: str,
    prompt: Dict[str, str],
    previous_errors: Optional[str] = None,
    previous_error_query: Optional[str] = None
):
    """
    Augment datasource metadata with additional information and format as JSON.

    This function retrieves the data dictionary and sample field values for a given
    datasource, adds them to the provided prompt dictionary, and includes any previous
    errors or queries for debugging purposes.

    Args:
        api_key (str): The API key for authentication.
        url (str): The base URL for the API endpoints.
        datasource_luid (str): The unique identifier of the datasource.
        prompt (Dict[str, str]): Initial prompt dictionary to be augmented.
        previous_errors (Optional[str]): Any errors from previous function calls. Defaults to None.
        previous_error_query (Optional[str]): The query that caused errors in previous calls. Defaults to None.

    Returns:
        str: A JSON string containing the augmented prompt dictionary with datasource metadata.

    Note:
        This function relies on external functions `get_data_dictionary` and `query_vds_metadata`
        to retrieve the necessary datasource information.
    """
    #  get sample values for fields from VDS metadata endpoint
    datasource_metadata = query_vds_metadata(
        api_key=api_key,
        url=url,
        datasource_luid=datasource_luid
    )

    for field in datasource_metadata['data']:
        del field['fieldName']
        del field['logicalTableId']

    prompt['data_model'] = datasource_metadata['data']

    # include previous error and query to debug in current run
    if previous_errors:
        prompt['previous_call_error'] = previous_errors
    if previous_error_query:
        prompt['previous_error_query'] = previous_error_query

    return json.dumps(prompt)


def prepare_prompt_inputs(data: dict, user_string: str) -> dict:
    """
    Prepare inputs for the prompt template with explicit, safe mapping.

    Args:
        data (dict): Raw data from VizQL query
        user_input (str): Original user query

    Returns:
        dict: Mapped inputs for PromptTemplate
    """
    return {
        "vds_query": data.get('query', ''),
        "data_source": data.get('data_source', ''),
        "data_table": data.get('data_table', ''),
        "user_input": user_string
    }


def env_vars_datasource_qa(
    domain=None,
    site=None,
    jwt_client_id=None,
    jwt_secret_id=None,
    jwt_secret=None,
    tableau_api_version=None,
    tableau_user=None,
    datasource_luid=None,
    tooling_llm_model=None
):
    """
    Retrieves Tableau configuration from environment variables if not provided as arguments.

    Args:
        domain (str, optional): Tableau domain
        site (str, optional): Tableau site
        jwt_client_id (str, optional): JWT client ID
        jwt_secret_id (str, optional): JWT secret ID
        jwt_secret (str, optional): JWT secret
        tableau_api_version (str, optional): Tableau API version
        tableau_user (str, optional): Tableau user
        datasource_luid (str, optional): Datasource LUID
        tooling_llm_model (str, optional): Tooling LLM model

    Returns:
        dict: A dictionary containing all the configuration values
    """
    # Load environment variables before accessing them
    load_dotenv()

    config = {
        'domain': domain if isinstance(domain, str) and domain else os.environ['TABLEAU_DOMAIN'],
        'site': site or os.environ['TABLEAU_SITE'],
        'jwt_client_id': jwt_client_id or os.environ['TABLEAU_JWT_CLIENT_ID'],
        'jwt_secret_id': jwt_secret_id or os.environ['TABLEAU_JWT_SECRET_ID'],
        'jwt_secret': jwt_secret or os.environ['TABLEAU_JWT_SECRET'],
        'tableau_api_version': tableau_api_version or os.environ['TABLEAU_API_VERSION'],
        'tableau_user': tableau_user or os.environ['TABLEAU_USER'],
        'datasource_luid': datasource_luid or os.environ['DATASOURCE_LUID'],
        'tooling_llm_model': tooling_llm_model or os.environ['TOOLING_MODEL']
    }

    return config


def query_vds(api_key: str, datasource_luid: str, url: str, query: Dict[str, Any]) -> Dict[str, Any]:
    full_url = f"{url}/api/v1/vizql-data-service/query-datasource"

    payload = {
        "datasource": {
            "datasourceLuid": datasource_luid
        },
        "query": query
    }

    headers = {
        'X-Tableau-Auth': api_key,
        'Content-Type': 'application/json'
    }

    response = requests.post(full_url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        error_message = (
            f"Failed to query data source via Tableau VizQL Data Service. "
            f"Status code: {response.status_code}. Response: {response.text}"
        )
        raise RuntimeError(error_message)


def query_vds_metadata(api_key: str, datasource_luid: str, url: str) -> Dict[str, Any]:
    full_url = f"{url}/api/v1/vizql-data-service/read-metadata"

    payload = {
        "datasource": {
            "datasourceLuid": datasource_luid
        }
    }

    headers = {
        'X-Tableau-Auth': api_key,
        'Content-Type': 'application/json'
    }

    response = requests.post(full_url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        error_message = (
            f"Failed to obtain data source metadata from VizQL Data Service. "
            f"Status code: {response.status_code}. Response: {response.text}"
        )
        raise RuntimeError(error_message)


def json_to_markdown_table(json_data):
    if isinstance(json_data, str):
        json_data = json.loads(json_data)
    # Check if the JSON data is a list and not empty
    if not isinstance(json_data, list) or not json_data:
        raise ValueError(f"Invalid JSON data, you may have an error or if the array is empty then it was not possible to resolve the query your wrote: {json_data}")

    headers = json_data[0].keys()

    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "| " + " | ".join(['---'] * len(headers)) + " |\n"

    for entry in json_data:
        row = "| " + " | ".join(str(entry[header]) for header in headers) + " |"
        markdown_table += row + "\n"

    return markdown_table
