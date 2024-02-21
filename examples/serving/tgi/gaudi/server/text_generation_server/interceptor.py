import torch
import grpc

from google.rpc import status_pb2, code_pb2
from grpc_status import rpc_status
from grpc_interceptor.server import AsyncServerInterceptor
from loguru import logger
from typing import Callable, Any
import traceback
import os


class ExceptionInterceptor(AsyncServerInterceptor):
    async def intercept(
        self,
        method: Callable,
        request_or_iterator: Any,
        context: grpc.ServicerContext,
        method_name: str,
    ) -> Any:
        try:
            response = method(request_or_iterator, context)
            return await response
        except Exception as err:
            trace = " " + traceback.format_exc() if os.environ.get('DUMP_STACK') else ''
            method_name = method_name.split("/")[-1]
            logger.exception(f"Method {method_name} encountered an error.")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            await context.abort_with_status(
                rpc_status.to_status(
                    status_pb2.Status(code=code_pb2.INTERNAL, message=str(err) + trace)
                )
            )
