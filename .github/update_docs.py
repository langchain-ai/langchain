import os
import re

# Directories containing the model implementations and the documentation
code_dir = "libs/langchain/langchain/llms"
docs_dir = "docs/extras/integrations/llms"

# Regex pattern to find the 'if self.streaming:' line
pattern = re.compile(r"if self\.streaming:")

# Placeholder in the documentation files
placeholder = "{{ streaming_support }}"

# Iterate over all Python files in the code directory
for filename in os.listdir(code_dir):
    if filename.endswith(".py"):
        with open(os.path.join(code_dir, filename)) as file:
            # Check if 'if self.streaming:' exists in the file
            streaming_support = "✅" if pattern.search(file.read()) else "❌"
            
            # Update the corresponding documentation file
            doc_filename = filename[:-3] + ".md"  # Replace .py with .md
            doc_filepath = os.path.join(docs_dir, doc_filename)
            if os.path.exists(doc_filepath):
                with open(doc_filepath, "r+") as doc_file:  # Open in read/write mode
                    doc_content = doc_file.read()
                    doc_content = doc_content.replace(placeholder, streaming_support)
                    doc_file.seek(0)  # Go back to the start of the file
                    doc_file.write(doc_content)
                    doc_file.truncate()  # Remove any remaining old content
            else:
                print(f"Warning: Documentation file {doc_filename} does not exist.")
