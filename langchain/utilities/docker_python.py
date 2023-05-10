import io
import tarfile

import docker


class DockerPythonREPL:
    """Simulates a Python REPL inside a Docker container."""

    def __init__(self):
        self.client = docker.from_env()
        self.container = self.client.containers.run(
            "python:3.9", tty=True, detach=True, auto_remove=True
        )

    def run(self, command: str) -> str:
        """Create and run Python file in the Docker container."""
        # Create the Python code.
        python_code = f"print({command})"

        # Escape double quotes in the Python code.
        python_code = python_code.replace('"', '\\"')

        # Write the Python code to a file inside the Docker container.
        result = self.container.exec_run(
            f'bash -c "echo -e \\"{python_code}\\" > /temp.py"'
        )

        # Execute the Python file in the Docker container.
        result = self.container.exec_run("python /temp.py")
        return result.output.decode("utf-8")

    def stop(self) -> None:
        self.container.stop()


if __name__ == "__main__":
    import time

    # Instantiate the DockerPythonREPL.
    repl = DockerPythonREPL()

    # Run some Python code.
    command = "2 + 2"
    output = repl.run(command)
    print(f">>> {command}\n>>>", output)

    # Run some more Python code.
    command = "'hello, ' + 'world!'"
    output = repl.run(command)
    print(f">>> {command}\n>>>", output)

    # Don't forget to stop the container when you're done with it!
    print("Stopping container...")
    t0 = time.time()
    repl.stop()
    print(f"Container stopped in {time.time()-t0} seconds.")
