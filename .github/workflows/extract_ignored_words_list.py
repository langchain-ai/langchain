import toml

pyproject_toml = toml.load("./lib/langchain/pyproject.toml")

# Extract the ignore words list (adjust the key as per your TOML structure)
ignore_words_list = pyproject_toml.get("tool", {}).get("codespell", {}).get("ignore_words_list")

print(f"::set-output name=ignore_words_list::{ignore_words_list}")