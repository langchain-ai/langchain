import logging
import os

log = logging.getLogger(__name__)


def get_user_agent() -> str:
    """Get user agent from environment variable."""
    env_user_agent = os.environ.get("USER_AGENT")
    if not env_user_agent:
        log.warning(
            "USER_AGENT environment variable not set, "
            "consider setting it to identify your requests."
        )
        return "DefaultLangchainUserAgent"
    return env_user_agent
