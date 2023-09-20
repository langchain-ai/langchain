"""Implementation of the server which will be copied to docker container.
You should not run it for other purposes than testing.
"""
# CAREFUL! This file is copied to docker container and executed there.
# it is based only on python standard library and pydantic as other dependencies
# are not available in the container. See python_docker_repl.py for more info
# and check _get_dockerfile_content() function.
import ast
import http.server
import sys
from contextlib import redirect_stdout
from enum import Enum
from io import StringIO
from typing import Dict, Optional, Union

from pydantic import BaseModel, ValidationError


class CommandName(Enum):
    RESET = "reset"
    QUIT = "quit"


class Cmd(BaseModel):
    cmd: CommandName

    class Config:
        extra = "forbid"


class Code(BaseModel):
    code: str
    use_ast: bool = False

    class Config:
        extra = "forbid"


class InputMessage(BaseModel):
    __root__: Union[Code, Cmd]


class OutputMessage(BaseModel):
    result: str

    class Config:
        extra = "forbid"


REPL_GLOBALS = {}
REPL_LOCALS = {}


def _run_ast(
    code: str, globals: Optional[Dict] = None, locals: Optional[Dict] = None
) -> str:
    """It execs code with intention of capturing the result of the last line,
    similar to output of python REPL if you type a command.
    """
    tree = ast.parse(code)
    module = ast.Module(tree.body[:-1], type_ignores=[])
    exec(ast.unparse(module), globals, locals)  # type: ignore
    module_end = ast.Module(tree.body[-1:], type_ignores=[])
    module_end_str = ast.unparse(module_end)  # type: ignore
    io_buffer = StringIO()
    try:
        with redirect_stdout(io_buffer):
            ret = eval(module_end_str, globals, locals)
            if ret is None:
                return io_buffer.getvalue()
            else:
                return ret
    except Exception:
        with redirect_stdout(io_buffer):
            exec(module_end_str, globals, locals)
        return io_buffer.getvalue()


def run_ast(
    code: str, globals: Optional[Dict] = None, locals: Optional[Dict] = None
) -> str:
    """It is a wrapper around _run_ast that catches exceptions so it
    behaves as run_code.
    """
    try:
        return _run_ast(code, globals, locals)
    except Exception as ex:
        return repr(ex.with_traceback(None))


def run_code(
    code: str, globals: Optional[Dict] = None, locals: Optional[Dict] = None
) -> str:
    """Executes code and returns captured stdout. or error message."""
    old_stdout = sys.stdout
    sys.stdout = new_stdout = StringIO()
    try:
        exec(code, globals, locals)
        return new_stdout.getvalue()
    except Exception as ex:
        return repr(ex.with_traceback(None))
    finally:
        sys.stdout = old_stdout


class PythonREPLService(http.server.BaseHTTPRequestHandler):
    """Simple python REPL server - http server that accepts json requests and
    returns json responses.
    NOTE: this object is created for each request so it is stateless.
    """

    def do_GET(self):
        self.send_response(200, "OK")
        self.end_headers()
        self.wfile.write(b"Hello! I am a python REPL server.")

    def do_POST(self):
        length = int(self.headers.get("content-length"))
        data = self.rfile.read(length)

        try:
            cmd = InputMessage.parse_raw(data)
        except ValidationError as exc:
            self.send_response(400, "Bad Request")
            self.end_headers()
            self.wfile.write(exc.json().encode("utf-8"))
            self.close_connection = True
            return

        cmd = cmd.__root__
        global REPL_LOCALS
        global REPL_GLOBALS
        if isinstance(cmd, Cmd):
            if cmd.cmd == CommandName.QUIT:
                self.send_response(200, "OK")
                self.end_headers()
                self.wfile.write(b"")
                # to kill, we send CTRL_C_EVENT to our own process.
                import os
                import signal

                os.kill(os.getpid(), signal.CTRL_C_EVENT)
            elif cmd.cmd == CommandName.RESET:
                REPL_GLOBALS = {}
                REPL_LOCALS = {}
                self.send_response(200, "OK")
                self.end_headers()
                self.wfile.write(b"")
        elif isinstance(cmd, Code):
            executor = run_ast if cmd.use_ast else run_code
            # NOTE: we only pass globals, otherwise for example code like this:
            # def f():
            #    return 42
            # print(f())
            # would not work.
            result = str(executor(cmd.code, REPL_GLOBALS))
            self.send_response(200, "OK")
            self.end_headers()
            self.wfile.write(OutputMessage(result=result).json().encode("utf-8"))
        else:
            self.send_response(400, "Bad Request")
            self.end_headers()
            self.wfile.write(b"Failed to parse input.")


def run_server() -> None:
    """Runs http server that accepts json requests and returns json responses."""
    http.server.HTTPServer.allow_reuse_address = True
    # NOTE: 8080 is internal and important hardcoded port number to match
    # python_docker_repl.py, do not change it.
    httpd = http.server.HTTPServer(("", 8080), PythonREPLService)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()


def run() -> None:
    """Runs infinite loop of python REPL."""
    run_server()


if __name__ == "__main__":
    run()
