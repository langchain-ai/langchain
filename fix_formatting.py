#!/usr/bin/env python3
with open('libs/core/langchain_core/utils/json_schema.py', 'r') as f:
    lines = f.readlines()

# Find and fix the problematic line
for i, line in enumerate(lines):
    if 'isinstance(out, list) and 0 <= index < len(out)) or (isinstance(out, dict) and index in out):' in line:
        # Replace with properly formatted if/elif blocks
        lines[i] = '            if isinstance(out, list) and 0 <= index < len(out):\n'
        lines.insert(i+1, '                out = out[index]\n')
        lines.insert(i+2, '            elif isinstance(out, dict) and index in out:\n')
        break

with open('libs/core/langchain_core/utils/json_schema.py', 'w') as f:
    f.writelines(lines)

print('Fixed formatting')
