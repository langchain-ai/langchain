import base64
import mimetypes


def encode_image(image_path: str) -> str:
    """Get base64 string from image URI."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_to_data_url(image_path: str) -> str:
    encoding = encode_image(image_path)
    mime_type = mimetypes.guess_type(image_path)[0]
    return f"data:{mime_type};base64,{encoding}"
