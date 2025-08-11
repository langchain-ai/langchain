with open('langchain_openai/chat_models/base.py', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "Uses OpenAI's Batch API for 50% cost savings" in line:
        lines[i] = line.replace(
            "Uses OpenAI's Batch API for 50% cost savings",
            "Uses OpenAI's Batch API\n           for 50% cost savings"
        )
        break

with open('langchain_openai/chat_models/base.py', 'w') as f:
    f.writelines(lines)
