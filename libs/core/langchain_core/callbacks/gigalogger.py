import os
import threading
import uuid
from typing import Any

GIGALOGGER_HOST = "https://gigalogger.demo.sberdevices.ru"

# Глобальные переменные, чтобы не инициализировать logger несколько раз за старт проекта
INITIALIZED = False
HANDLER = None

init_lock = threading.Lock()


class GigaLoggerInitializeException(Exception):
    def __init__(self, message: str) -> None:
        self.message = message


def create_gigalogger_handler() -> Any:
    # Этот метод пытается подключиться только один раз при старте проекта к гигалоггеру
    # Если у него это не выходит, он не повторяет попытки при инициализации других
    # частей цепочки
    global HANDLER, INITIALIZED
    with init_lock:
        if INITIALIZED:
            return HANDLER
        try:
            from langfuse.callback import (  # type: ignore[import-untyped]
                CallbackHandler as LangFuseCallback,
            )
        except ImportError as e:
            raise ImportError(
                "Could not import langfuse python package. "
                "For correct work of gigalogger langfuse is required. "
                "Please install it with `pip install langfuse`."
            ) from e

        try:
            pk = os.environ["GIGALOGGER_PUBLIC_KEY"]
            sk = os.environ["GIGALOGGER_SECRET_KEY"]
        except KeyError as e:
            INITIALIZED = True
            raise GigaLoggerInitializeException(
                "Set 'GIGALOGGER_PUBLIC_KEY' and 'GIGALOGGER_SECRET_KEY' "
                "environment variables."
            ) from e
        HANDLER = LangFuseCallback(
            public_key=pk,
            secret_key=sk,
            host=os.environ.get("GIGALOGGER_HOST") or GIGALOGGER_HOST,
            session_id=os.environ.get("GIGALOGGER_SESSION_ID") or str(uuid.uuid4()),
        )
        try:
            HANDLER.auth_check()
        except Exception as e:
            HANDLER = None
            raise GigaLoggerInitializeException(
                "Failed to authenticate in GigaLogger. "
                "Check your public and secret key. "
                f"Additional message: '{repr(e)}'"
            ) from e
        finally:
            INITIALIZED = True
        return HANDLER


def _gigalogger_is_enabled() -> bool:
    return os.environ.get("GIGALOGGER_ENABLED", "").lower() == "true"
