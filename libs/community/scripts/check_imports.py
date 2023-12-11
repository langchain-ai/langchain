import sys
from importlib.machinery import SourceFileLoader

if __name__ == "__main__":
    files = sys.argv[1:]
    failures = []
    for file in files:
        try:
            SourceFileLoader("x", file).load_module()
        except Exception as e:
            failures.append((file, e))

    if not failures:
        sys.exit(0)

    for file, ex in failures:
        print(file)
        print(ex)

    sys.exit(1)
