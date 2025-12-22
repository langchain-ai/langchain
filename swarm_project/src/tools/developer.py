from langchain_core.tools import tool

@tool
def write_code(file_path: str, content: str):
    """
    Writes content to a file at the specified path.
    Useful for creating scripts, configuration files, or any text-based artifact.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing to {file_path}: {str(e)}"
