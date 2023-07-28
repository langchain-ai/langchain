import os
import re

# Directories containing the model implementations and the documentation
code_dir = "libs/langchain/langchain/llms"
docs_dir = "docs/extras/integrations/llms"

# Regex pattern to find the 'if self.streaming:' line
pattern = re.compile(r"if self\.streaming:")

# Iterate over all Python files in the code directory
for filename in os.listdir(code_dir):
    if filename.endswith(".py"):
        with open(os.path.join(code_dir, filename)) as file:
            # Check if 'if self.streaming:' exists in the file
            if pattern.search(file.read()):
                # If it does, update the corresponding documentation file
                doc_filename = filename[:-3] + ".md"  # Replace .py with .md
                doc_filepath = os.path.join(docs_dir, doc_filename)
                if os.path.exists(doc_filepath):
                    with open(doc_filepath, "a") as doc_file:  # Open in append mode
                        doc_file.write("\n\nModel currently supports streaming ✅\n")
                else:
                    print(f"Warning: Documentation file {doc_filename} does not exist.")
