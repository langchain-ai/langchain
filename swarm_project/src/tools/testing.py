from langchain_core.tools import tool
import subprocess

@tool
def run_test(command: str):
    """
    Runs a shell command to test the code.
    Returns the stdout and stderr.
    Useful for running python scripts (e.g., 'python script.py') or tests.
    """
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return f"PASS:\n{result.stdout}"
        else:
            return f"FAIL:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except Exception as e:
        return f"Execution Error: {str(e)}"
