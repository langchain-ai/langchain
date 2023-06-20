import os

import requests
from pydantic import BaseModel, Field

from langchain.tools import tool

#: Free tier available in RapidAPI : 20 k req/month

DOMAINR_API_KEY = os.environ["DOMAINR_API_KEY"]


class DomainAvailabilityRequest(BaseModel):
    domain: str = Field(description="Domain name (ex. reddit.com)")


@tool(
    "domain_availability",
    return_direct=True,
    args_schema=DomainAvailabilityRequest,
)
def domain_availability(domain: str) -> dict:
    """
    Retrieves the availability for a domain name using Domainr service
    """

    endpoint = "https://domainr.p.rapidapi.com/v2/status"

    params = {
        "mashape-key": DOMAINR_API_KEY,
        "domain": domain,
    }
    headers = {
        "X-RapidAPI-Key": DOMAINR_API_KEY,
        "X-RapidAPI-Host": "domainr.p.rapidapi.com",
    }

    response = requests.get(endpoint, headers=headers, params=params)
    return response.json()
