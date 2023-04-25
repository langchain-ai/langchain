"""
    -- @Time    : 2023/4/25 11:35
    -- @Author  : yazhui Yu
    -- @email   : yuyazhui@bangdao-tech.com
    -- @File    : network
    -- @Software: Pycharm
"""
import asyncio
import atexit
from typing import Any

import httpx

from .setting import HTTP_TIMEOUT

__session_pool = {}

HEADERS = {}


@atexit.register
def __clean() -> None:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        return

    async def __clean_task():
        await __session_pool[loop].close()

    if loop.is_closed():
        loop.run_until_complete(__clean_task())
    else:
        loop.create_task(__clean_task())


async def request(method: str, url: str, headers: dict) -> Any:
    config = {
        "method": method,
        "url": url,
        "headers": headers
    }

    session = get_session()
    resp = await session.request(**config)

    return resp


def get_session() -> httpx.AsyncClient:
    """
    获取当前模块的 httpx.AsyncClient 对象，用于自定义请求
    """
    global __session_pool
    loop = asyncio.get_event_loop()
    session = __session_pool.get(loop, None)
    if session is None:
        session = httpx.AsyncClient(timeout=HTTP_TIMEOUT)
        __session_pool[loop] = session

    return session
