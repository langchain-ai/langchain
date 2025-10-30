# Model Profile Augmentations

This directory contains LangChain-specific augmentations to the base model data from models.dev.

## Structure

```
augmentations/
├── providers/          # Provider-level augmentations (apply to all models from a provider)
│   ├── anthropic.toml
│   ├── openai.toml
│   └── ...
└── models/            # Model-specific augmentations (override provider defaults)
    ├── anthropic/
    │   └── claude-sonnet-4-5-20250929.toml
    └── openai/
        └── o1.toml
```

## Merge Priority

Data is merged in the following order (later overrides earlier):

1. Base data from `models.json` (from models.dev API)
2. Provider-level augmentations from `augmentations/providers/{provider}.toml`
3. Model-level augmentations from `augmentations/models/{provider}/{model}.toml`

## TOML Format

All augmentation files should use the following structure:

```toml
[profile]
# Add or override model profile fields
image_url_inputs = true
pdf_inputs = true
tool_choice = true
```

Available fields match the `ModelProfile` TypedDict in `model_profile.py`.
