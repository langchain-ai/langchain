from typing import Any


def __getattr__(name: str) -> Any:
    if name in ("encode_image", "image_to_data_url"):
        msg = f"'{name}' has been removed for security reasons."
        raise ValueError(msg)
    raise AttributeError(name)
