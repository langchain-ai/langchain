"""
    -- @Time    : 2023/4/25 9:27
    -- @Author  : yazhui Yu
    -- @email   : yuyazhui@bangdao-tech.com
    -- @File    : sync
    -- @Software: Pycharm
"""
from typing import Coroutine
import asyncio
from typing import Any


def __ensure_event_loop() -> None:
    # noinspection PyBroadException
    try:
        asyncio.get_event_loop()
    except:
        asyncio.set_event_loop(asyncio.new_event_loop())


def sync(coroutine: Coroutine) -> Any:
    __ensure_event_loop()
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coroutine)
