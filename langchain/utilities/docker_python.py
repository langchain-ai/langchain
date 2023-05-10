"""
Works!
"""

import docker
import time

class DockerPythonREPL:
    def __init__(self):
        self.client = docker.from_env()
        self.container = self.client.containers.run("python:3.9", tty=True, detach=True, auto_remove=True)

    # Version 1: write to host then copy to Docker container
    def run(self, command):
        # Write the command to a Python file.
        python_code = f"print({command})"
        with open("temp.py", "w") as file:
            file.write(python_code)

        # Copy the file to the Docker container.
        self.client.containers.get(self.container.id).put_archive(
            path="/",
            data=self._create_tarfile("temp.py")
        )

        # Execute the Python file in the Docker container.
        result = self.container.exec_run("python /temp.py")
        return result.output.decode('utf-8')

    # Version 2: directly create file in Docker container
    def run(self, command):
        # Create the Python code.
        python_code = f"print({command})"

        # Escape double quotes in the Python code.
        python_code = python_code.replace('"', '\\"')

        # Write the Python code to a file inside the Docker container.
        result = self.container.exec_run(f'bash -c "echo -e \\"{python_code}\\" > /temp.py"')

        # Execute the Python file in the Docker container.
        result = self.container.exec_run("python /temp.py")
        return result.output.decode('utf-8')


    def stop(self):
        self.container.stop()

    @staticmethod
    def _create_tarfile(filename):
        import tarfile
        import io

        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode='w') as tar:
            tar.add(filename)

        data.seek(0)
        return data

# Instantiate the DockerPythonREPL.
repl = DockerPythonREPL()

# Run some Python code.
output = repl.run("2 + 2")
print('Some code', output)

# Run some more Python code.
output = repl.run("'hello, ' + 'world!'")
print('More code', output)

# Don't forget to stop the container when you're done with it!
print('Stopping container...')
t0 = time.time()
repl.stop()
print(f'Container stopped in {time.time()-t0} seconds.')

