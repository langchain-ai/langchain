import asyncio
import base64
from io import BytesIO


def get_img_base64(file_name: str) -> str:
    """
    get_img_base64 used in streamlit.
    absolute local path not working on windows.
    """
    # 读取图片
    with open(file_name, "rb") as f:
        buffer = BytesIO(f.read())
        base_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{base_str}"


def ensure_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


# Create an event loop to run the async functions synchronously
def run_sync(func, *args, **kwargs):
    loop = ensure_event_loop()
    asyncio.set_event_loop(loop)

    return asyncio.run(func(*args, **kwargs))
