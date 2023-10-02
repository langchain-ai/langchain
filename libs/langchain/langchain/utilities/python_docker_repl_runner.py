"""Implementation of the server which will be copied to docker container.
You should not run it for other purposes than testing.
"""
# CAREFUL! This file is copied to docker container and executed there.
# it is based only on python standard library as other dependencies
# are not available in the container. See python_docker_repl.py for more info
# and check _get_dockerfile_content() function.

import ast
import http.server
import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Dict, Optional

REPL_GLOBALS: Dict[str, Any] = {}
REPL_LOCALS: Dict[str, Any] = {}


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

    def do_GET(self) -> None:
        self.send_response(200, "OK")
        self.end_headers()
        self.wfile.write(b"Hello! I am a python REPL server.")

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length"))
        data = self.rfile.read(length)

        try:
            cmd_json = json.loads(data)
        except json.JSONDecodeError as exc:
            self.send_response(400, "Bad Request")
            self.end_headers()
            self.wfile.write(f"Failed to parse input: {exc}".encode("utf-8"))
            self.close_connection = True
            return

        global REPL_LOCALS
        global REPL_GLOBALS

        if "code" in cmd_json:
            use_ast = cmd_json.get("use_ast", False)
            executor = run_ast if use_ast else run_code
            result = str(executor(cmd_json["code"], REPL_GLOBALS))
            self.send_response(200, "OK")
            self.end_headers()
            self.wfile.write(json.dumps({"result": result}).encode("utf-8"))

        else:
            self.send_response(400, "Bad Request")
            self.end_headers()
            self.wfile.write(b"Invalid input format.")


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
