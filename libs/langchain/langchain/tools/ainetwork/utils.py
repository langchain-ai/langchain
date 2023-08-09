"""AINetwork Blockchain tool utils."""
import logging
import os
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ain.ain import Ain

logger = logging.getLogger(__name__)


def authenticate() -> Union[Ain, None]:
    """Authenticate using the AIN Blockchain"""

    # Try importing ain-py related modules
    try:
        from ain.ain import Ain
    except ImportError as e:
        logger.error(
            "Cannot import ain-py related modules. Please install the package with `pip install ain-py`."
        )
        return None

    # Get PRIVATE_KEY from environment variables
    if (
        "AIN_BLOCKCHAIN_PROVIDER_URL" in os.environ
        and "AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY" in os.environ
    ):
        provider_url = os.environ["AIN_BLOCKCHAIN_PROVIDER_URL"]
        private_key = os.environ["AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY"]
    else:
        logger.error(
            "Error: The AIN_BLOCKCHAIN_PROVIDER_URL and AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY environmental variable has not been set."
        )
        return None

    # Initialize the AIN Account with the given private key
    try:
        ain = Ain(provider_url)
        ain.wallet.addAndSetDefaultAccount(private_key)
        return ain
    except Exception as e:
        logger.error(f"Error initializing AIN account: {str(e)}")
        return None
