import os
import requests
from pydantic import BaseModel, Field
from langchain.tools import tool


class EmailVerificationRequest(BaseModel):
    email: str = Field(description='Email to verify')


class EmailFinderRequest(BaseModel):
    full_name: str = Field(None, description='Full name (ex. John Appleseed)')
    domain: str = Field(description='Domain name (ex. reddit.com)')
    company: str = Field(description='Company name (ex. Reddit)')


@tool("search_email", return_direct=True, args_schema=EmailFinderRequest)
def search_email(full_name, domain, company) -> str:
    """
    Retrieves an email address using Hunter.io service
    """

    endpoint = 'https://api.hunter.io/v2/email-finder'

    params = {
        'api_key': os.environ['HUNTER_API_KEY'],
        'max_duration': 20,
        'full_name': full_name,
        'domain': domain,
        'company': company,
    }
    response = requests.get(endpoint, params=params)
    return response.json()


@tool("verify_email", return_direct=True, args_schema=EmailVerificationRequest)
def verify_email(email) -> str:
    """
    Verifies an email address validity using Hunter.io service
    """

    endpoint = 'https://api.hunter.io/v2/email-finder'

    params = {
        'api_key': os.environ['HUNTER_API_KEY'],
        'max_duration': 20,
        'email': email,
    }
    response = requests.get(endpoint, params=params)
    return response.json()
