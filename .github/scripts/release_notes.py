from pathlib import Path
import sys

if __name__ == "__main__":
    # unpack args
    lib_path = Path(sys.argv[1])
    pyproject_path = lib_path / "pyproject.toml"
