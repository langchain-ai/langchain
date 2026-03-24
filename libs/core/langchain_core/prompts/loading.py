"""Load prompts."""

import json
import logging
from collections.abc import Callable
from pathlib import Path

import yaml

from langchain_core._api import deprecated
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

URL_BASE = "https://raw.githubusercontent.com/hwchase17/langchain-hub/master/prompts/"
logger = logging.getLogger(__name__)


def _validate_path(path: Path) -> None:
    """Reject absolute paths and ``..`` traversal components.

    Args:
        path: The path to validate.

    Raises:
        ValueError: If the path is absolute or contains ``..`` components.
    """
    if path.is_absolute():
        msg = (
            f"Path '{path}' is absolute. Absolute paths are not allowed "
            f"when loading prompt configurations to prevent path traversal "
            f"attacks. Use relative paths instead, or pass "
            f"`allow_dangerous_paths=True` if you trust the input."
        )
        raise ValueError(msg)
    if ".." in path.parts:
        msg = (
            f"Path '{path}' contains '..' components. Directory traversal "
            f"sequences are not allowed when loading prompt configurations. "
            f"Use direct relative paths instead, or pass "
            f"`allow_dangerous_paths=True` if you trust the input."
        )
        raise ValueError(msg)


@deprecated(
    since="1.2.21",
    removal="2.0.0",
    alternative="Use `dumpd`/`dumps` from `langchain_core.load` to serialize "
    "prompts and `load`/`loads` to deserialize them.",
)
def load_prompt_from_config(
    config: dict, *, allow_dangerous_paths: bool = False
) -> BasePromptTemplate:
    """Load prompt from config dict.

    Args:
        config: Dict containing the prompt configuration.
        allow_dangerous_paths: If ``False`` (default), file paths in the
            config (such as ``template_path``, ``examples``, and
            ``example_prompt_path``) are validated to reject absolute paths
            and directory traversal (``..``) sequences. Set to ``True`` only
            if you trust the source of the config.

    Returns:
        A `PromptTemplate` object.

    Raises:
        ValueError: If the prompt type is not supported.
    """
    if "_type" not in config:
        logger.warning("No `_type` key found, defaulting to `prompt`.")
    config_type = config.pop("_type", "prompt")

    if config_type not in type_to_loader_dict:
        msg = f"Loading {config_type} prompt not supported"
        raise ValueError(msg)

    prompt_loader = type_to_loader_dict[config_type]
    return prompt_loader(config, allow_dangerous_paths=allow_dangerous_paths)


def _load_template(
    var_name: str, config: dict, *, allow_dangerous_paths: bool = False
) -> dict:
    """Load template from the path if applicable."""
    # Check if template_path exists in config.
    if f"{var_name}_path" in config:
        # If it does, make sure template variable doesn't also exist.
        if var_name in config:
            msg = f"Both `{var_name}_path` and `{var_name}` cannot be provided."
            raise ValueError(msg)
        # Pop the template path from the config.
        template_path = Path(config.pop(f"{var_name}_path"))
        if not allow_dangerous_paths:
            _validate_path(template_path)
        # Load the template.
        if template_path.suffix == ".txt":
            template = template_path.read_text(encoding="utf-8")
        else:
            raise ValueError
        # Set the template variable to the extracted variable.
        config[var_name] = template
    return config


def _load_examples(config: dict, *, allow_dangerous_paths: bool = False) -> dict:
    """Load examples if necessary."""
    if isinstance(config["examples"], list):
        pass
    elif isinstance(config["examples"], str):
        path = Path(config["examples"])
        if not allow_dangerous_paths:
            _validate_path(path)
        with path.open(encoding="utf-8") as f:
            if path.suffix == ".json":
                examples = json.load(f)
            elif path.suffix in {".yaml", ".yml"}:
                examples = yaml.safe_load(f)
            else:
                msg = "Invalid file format. Only json or yaml formats are supported."
                raise ValueError(msg)
        config["examples"] = examples
    else:
        msg = "Invalid examples format. Only list or string are supported."
        raise ValueError(msg)  # noqa:TRY004
    return config


def _load_output_parser(config: dict) -> dict:
    """Load output parser."""
    if config_ := config.get("output_parser"):
        if output_parser_type := config_.get("_type") != "default":
            msg = f"Unsupported output parser {output_parser_type}"
            raise ValueError(msg)
        config["output_parser"] = StrOutputParser(**config_)
    return config


def _load_few_shot_prompt(
    config: dict, *, allow_dangerous_paths: bool = False
) -> FewShotPromptTemplate:
    """Load the "few shot" prompt from the config."""
    # Load the suffix and prefix templates.
    config = _load_template(
        "suffix", config, allow_dangerous_paths=allow_dangerous_paths
    )
    config = _load_template(
        "prefix", config, allow_dangerous_paths=allow_dangerous_paths
    )
    # Load the example prompt.
    if "example_prompt_path" in config:
        if "example_prompt" in config:
            msg = (
                "Only one of example_prompt and example_prompt_path should "
                "be specified."
            )
            raise ValueError(msg)
        example_prompt_path = Path(config.pop("example_prompt_path"))
        if not allow_dangerous_paths:
            _validate_path(example_prompt_path)
        config["example_prompt"] = load_prompt(
            example_prompt_path, allow_dangerous_paths=allow_dangerous_paths
        )
    else:
        config["example_prompt"] = load_prompt_from_config(
            config["example_prompt"], allow_dangerous_paths=allow_dangerous_paths
        )
    # Load the examples.
    config = _load_examples(config, allow_dangerous_paths=allow_dangerous_paths)
    config = _load_output_parser(config)
    return FewShotPromptTemplate(**config)


def _load_prompt(
    config: dict, *, allow_dangerous_paths: bool = False
) -> PromptTemplate:
    """Load the prompt template from config."""
    # Load the template from disk if necessary.
    config = _load_template(
        "template", config, allow_dangerous_paths=allow_dangerous_paths
    )
    config = _load_output_parser(config)

    template_format = config.get("template_format", "f-string")
    if template_format == "jinja2":
        # Disabled due to:
        # https://github.com/langchain-ai/langchain/issues/4394
        msg = (
            f"Loading templates with '{template_format}' format is no longer supported "
            f"since it can lead to arbitrary code execution. Please migrate to using "
            f"the 'f-string' template format, which does not suffer from this issue."
        )
        raise ValueError(msg)

    return PromptTemplate(**config)


@deprecated(
    since="1.2.21",
    removal="2.0.0",
    alternative="Use `dumpd`/`dumps` from `langchain_core.load` to serialize "
    "prompts and `load`/`loads` to deserialize them.",
)
def load_prompt(
    path: str | Path,
    encoding: str | None = None,
    *,
    allow_dangerous_paths: bool = False,
) -> BasePromptTemplate:
    """Unified method for loading a prompt from LangChainHub or local filesystem.

    Args:
        path: Path to the prompt file.
        encoding: Encoding of the file.
        allow_dangerous_paths: If ``False`` (default), file paths referenced
            inside the loaded config (such as ``template_path``, ``examples``,
            and ``example_prompt_path``) are validated to reject absolute paths
            and directory traversal (``..``) sequences. Set to ``True`` only
            if you trust the source of the config.

    Returns:
        A `PromptTemplate` object.

    Raises:
        RuntimeError: If the path is a LangChainHub path.
    """
    if isinstance(path, str) and path.startswith("lc://"):
        msg = (
            "Loading from the deprecated github-based Hub is no longer supported. "
            "Please use the new LangChain Hub at https://smith.langchain.com/hub "
            "instead."
        )
        raise RuntimeError(msg)
    return _load_prompt_from_file(
        path, encoding, allow_dangerous_paths=allow_dangerous_paths
    )


def _load_prompt_from_file(
    file: str | Path,
    encoding: str | None = None,
    *,
    allow_dangerous_paths: bool = False,
) -> BasePromptTemplate:
    """Load prompt from file."""
    # Convert file to a Path object.
    file_path = Path(file)
    # Load from either json or yaml.
    if file_path.suffix == ".json":
        with file_path.open(encoding=encoding) as f:
            config = json.load(f)
    elif file_path.suffix.endswith((".yaml", ".yml")):
        with file_path.open(encoding=encoding) as f:
            config = yaml.safe_load(f)
    else:
        msg = f"Got unsupported file type {file_path.suffix}"
        raise ValueError(msg)
    # Load the prompt from the config now.
    return load_prompt_from_config(config, allow_dangerous_paths=allow_dangerous_paths)


def _load_chat_prompt(
    config: dict,
    *,
    allow_dangerous_paths: bool = False,  # noqa: ARG001
) -> ChatPromptTemplate:
    """Load chat prompt from config."""
    messages = config.pop("messages")
    template = messages[0]["prompt"].pop("template") if messages else None
    config.pop("input_variables")

    if not template:
        msg = "Can't load chat prompt without template"
        raise ValueError(msg)

    return ChatPromptTemplate.from_template(template=template, **config)


type_to_loader_dict: dict[str, Callable[..., BasePromptTemplate]] = {
    "prompt": _load_prompt,
    "few_shot": _load_few_shot_prompt,
    "chat": _load_chat_prompt,
}
