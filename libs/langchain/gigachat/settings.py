import logging
from base64 import b64encode
from typing import Any, Dict, Optional

from langchain.pydantic_v1 import BaseSettings, root_validator

ENV_PREFIX = "GIGA_"

# API_BASE_URL = "https://beta.saluteai.sberdevices.ru/v1"
API_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"

OAUTH_BASE_URL = "https://ngw.devices.sberbank.ru:9443/api/v2"
OAUTH_SCOPE = "GIGACHAT_API_CORP"


def _build_oauth_token(client_id: str, client_secret: str) -> str:
    return b64encode(f"{client_id}:{client_secret}".encode()).decode()


class Settings(BaseSettings):
    verbose: bool = False
    use_auth: bool = True

    api_base_url: str = API_BASE_URL
    token: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    model: Optional[str] = None
    timeout: float = 30.0
    verify_ssl: bool = True

    client_id: Optional[str] = None
    client_secret: Optional[str] = None

    oauth_base_url: str = OAUTH_BASE_URL
    oauth_token: Optional[str] = None
    oauth_scope: str = OAUTH_SCOPE
    oauth_timeout: float = 5.0
    oauth_verify_ssl: bool = True

    class Config:
        env_prefix = ENV_PREFIX

    @root_validator
    def compute_oauth_token(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not values["oauth_token"]:
            client_id, client_secret = (
                values["client_id"],
                values["client_secret"],
            )
            if client_id and client_secret:
                values["oauth_token"] = _build_oauth_token(client_id, client_secret)

        if values["use_auth"]:
            use_secrets = (
                values["oauth_token"]
                or values["token"]
                or (values["user"] and values["password"])
            )
            if not use_secrets:
                logging.warning(
                    "Please provide GIGA_CLIENT_ID and GIGA_CLIENT_SECRET"
                    " environment variables."
                )

        return values
