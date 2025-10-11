# Middleware Hooks Visualizer

A clean, interactive visualizer showing how middleware hooks affect LangChain agent graphs.

## Usage

### View the Application

Open `index.html` in a browser or serve via HTTP:

```bash
cd docs/middleware_diagrams
python -m http.server 8000
# Visit http://localhost:8000
```

### Using with Mintlify

The HTML is designed to be embedded directly into Mintlify docs. Simply include the HTML file or copy the relevant sections into your documentation.

## Binary Naming Scheme

Diagrams are named using a 7-bit binary string:

```
Bit 0: has_tools (1 = tools enabled, 0 = no tools)
Bit 1: before_agent hook
Bit 2: before_model hook
Bit 3: after_model hook
Bit 4: after_agent hook
Bit 5: wrap_model_call hook
Bit 6: wrap_tool_call hook
```

**Example**: `1010100` = tools enabled, before_model hook, after_agent hook

## Regenerating Diagrams

To regenerate all 128 diagram combinations:

```bash
cd libs/langchain_v1
uv run python scripts/generate_middleware_diagrams.py
```

This generates `diagrams.json` with all possible hook combinations.

## Files

- `index.html` - Main application (diagram + toggles)
- `diagrams.json` - 128 pre-generated Mermaid diagrams
- `scripts/generate_middleware_diagrams.py` - Diagram generator

## Design

- **Clean layout**: Diagram on left, toggles on right
- **LangChain colors**: Purple (#7c3aed) for accents, grey for backgrounds
- **Simple toggles**: Clean on/off switches for each hook
- **Instant rendering**: Mermaid.js renders diagrams in real-time
