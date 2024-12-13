from typing import Any


def __getattr__(name: str) -> Any:
    if name in ("encode_image", "image_to_data_url"):
        msg = (
            f"'{name}' has been removed for security reasons.\n\n"
            f"Usage of this utility in environments with user-input paths is a "
            f"security vulnerability. Out of an abundance of caution, the utility "
            f"has been removed to prevent possible misuse."
        )
        raise ValueError(msg)
    raise AttributeError(name)
