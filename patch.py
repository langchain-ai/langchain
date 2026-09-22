import re

with open('libs/core/langchain_core/language_models/_compat_bridge.py', 'r') as f:
    content = f.read()

old_logic = """        if explicit_idx is None:
            # No source-side identity. Bucket by (sentinel, block type,
            # positional `i`) so two blocks of different types at the
            # same position across chunks (e.g. Gemini emitting a
            # reasoning block in one chunk and a `tool_call` in the
            # next, both at positional 0 because each chunk carries one
            # block) get distinct wire blocks. Without this, the second
            # type's incoming block hits `_accumulate`'s self-contained
            # `else` branch and clobbers the first. Same-type chunks
            # still share the bucket and merge cleanly, which is what
            # streaming text / reasoning relies on.
            key: Any = ("__lc_no_index__", block.get("type"), i)
        else:
            key = explicit_idx"""

new_logic = """        if explicit_idx is None:
            # No source-side identity. Bucket by (sentinel, block type,
            # positional `i`) so two blocks of different types at the
            # same position across chunks (e.g. Gemini emitting a
            # reasoning block in one chunk and a `tool_call` in the
            # next, both at positional 0 because each chunk carries one
            # block) get distinct wire blocks. Without this, the second
            # type's incoming block hits `_accumulate`'s self-contained
            # `else` branch and clobbers the first. Same-type chunks
            # still share the bucket and merge cleanly, which is what
            # streaming text / reasoning relies on.
            if block.get("type") == "tool_call" and block.get("id"):
                key: Any = ("__lc_tool_call__", block["id"])
            else:
                key = ("__lc_no_index__", block.get("type"), i)
        else:
            key = explicit_idx"""

content = content.replace(old_logic, new_logic)

with open('libs/core/langchain_core/language_models/_compat_bridge.py', 'w') as f:
    f.write(content)
