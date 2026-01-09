import traceback
from pathlib import Path
from typing import Any

from langchain_prompty.core import (
    Frontmatter,
    InvokerFactory,
    ModelSettings,
    Prompty,
    PropertySettings,
    SimpleModel,
    TemplateSettings,
    param_hoisting,
)


def load(prompt_path: str, configuration: str = "default") -> Prompty:
    """Load a prompty file and return a Prompty object.

    Args:
        prompt_path: The path to the prompty file.
        configuration: The configuration to use. Defaults to `'default'`.

    Returns:
        The Prompty object.
    """
    file_path = Path(prompt_path)
    if not file_path.is_absolute():
        # get caller's path (take into account trace frame)
        caller = Path(traceback.extract_stack()[-3].filename)
        file_path = Path(caller.parent / file_path).resolve().absolute()

    # load dictionary from prompty file
    matter = Frontmatter.read_file(file_path.__fspath__())
    attributes = matter["attributes"]
    content = matter["body"]

    # normalize attribute dictionary resolve keys and files
    attributes = Prompty.normalize(attributes, file_path.parent)

    # load global configuration
    if "model" not in attributes:
        attributes["model"] = {}

    # pull model settings out of attributes
    try:
        model = ModelSettings(**attributes.pop("model"))
    except Exception as e:
        raise ValueError(f"Error in model settings: {e}")

    # pull template settings
    try:
        if "template" in attributes:
            t = attributes.pop("template")
            if isinstance(t, dict):
                template = TemplateSettings(**t)
            # has to be a string denoting the type
            else:
                template = TemplateSettings(type=t, parser="prompty")
        else:
            template = TemplateSettings(type="mustache", parser="prompty")
    except Exception as e:
        raise ValueError(f"Error in template loader: {e}")

    # formalize inputs and outputs
    if "inputs" in attributes:
        try:
            inputs = {
                k: PropertySettings(**v) for (k, v) in attributes.pop("inputs").items()
            }
        except Exception as e:
            raise ValueError(f"Error in inputs: {e}")
    else:
        inputs = {}
    if "outputs" in attributes:
        try:
            outputs = {
                k: PropertySettings(**v) for (k, v) in attributes.pop("outputs").items()
            }
        except Exception as e:
            raise ValueError(f"Error in outputs: {e}")
    else:
        outputs = {}

    # recursive loading of base prompty
    if "base" in attributes:
        # load the base prompty from the same directory as the current prompty
        base = load(file_path.parent / attributes["base"])
        # hoist the base prompty's attributes to the current prompty
        model.api = base.model.api if model.api == "" else model.api
        model.configuration = param_hoisting(
            model.configuration, base.model.configuration
        )
        model.parameters = param_hoisting(model.parameters, base.model.parameters)
        model.response = param_hoisting(model.response, base.model.response)
        attributes["sample"] = param_hoisting(attributes, base.sample, "sample")

        p = Prompty(
            **attributes,
            model=model,
            inputs=inputs,
            outputs=outputs,
            template=template,
            content=content,
            file=file_path,
            basePrompty=base,
        )
    else:
        p = Prompty(
            **attributes,
            model=model,
            inputs=inputs,
            outputs=outputs,
            template=template,
            content=content,
            file=file_path,
        )
    return p


def prepare(
    prompt: Prompty,
    inputs: dict[str, Any] = {},
) -> Any:
    """Prepare the inputs for the prompty.

    Args:
        prompt: The Prompty object.
        inputs: The inputs to the prompty. Defaults to `{}`.

    Returns:
        The prepared inputs.
    """
    invoker = InvokerFactory()

    inputs = param_hoisting(inputs, prompt.sample)

    if prompt.template.type == "NOOP":
        render = prompt.content
    else:
        # render
        result = invoker(
            "renderer",
            prompt.template.type,
            prompt,
            SimpleModel(item=inputs),
        )
        render = result.item

    if prompt.template.parser == "NOOP":
        result = render
    else:
        # parse
        result = invoker(
            "parser",
            f"{prompt.template.parser}.{prompt.model.api}",
            prompt,
            SimpleModel(item=result.item),
        )

    if isinstance(result, SimpleModel):
        return result.item
    else:
        return result


def run(
    prompt: Prompty,
    content: dict | list | str,
    configuration: dict[str, Any] = {},
    parameters: dict[str, Any] = {},
    raw: bool = False,
) -> Any:
    """Run the prompty.

    Args:
        prompt: The Prompty object.
        content: The content to run the prompty on.
        configuration: The configuration to use. Defaults to `{}`.
        parameters: The parameters to use. Defaults to `{}`.
        raw: Whether to return the raw output. Defaults to `False`.

    Returns:
        The result of running the prompty.
    """
    invoker = InvokerFactory()

    if configuration != {}:
        prompt.model.configuration = param_hoisting(
            configuration, prompt.model.configuration
        )

    if parameters != {}:
        prompt.model.parameters = param_hoisting(parameters, prompt.model.parameters)

    # execute
    result = invoker(
        "executor",
        prompt.model.configuration["type"],
        prompt,
        SimpleModel(item=content),
    )

    # skip?
    if not raw:
        # process
        result = invoker(
            "processor",
            prompt.model.configuration["type"],
            prompt,
            result,
        )

    if isinstance(result, SimpleModel):
        return result.item
    else:
        return result


def execute(
    prompt: str | Prompty,
    configuration: dict[str, Any] = {},
    parameters: dict[str, Any] = {},
    inputs: dict[str, Any] = {},
    raw: bool = False,
    connection: str = "default",
) -> Any:
    """Execute a `Prompty`.

    Args:
        prompt: The prompt to execute.
            Can be a path to a prompty file or a `Prompty` object.
        configuration: The configuration to use.
        parameters: The parameters to use.
        inputs: The inputs to the `Prompty`.
        raw: Whether to return the raw output.
        connection: The connection to use.

    Returns:
        The result of executing the `Prompty`.
    """

    if isinstance(prompt, str):
        prompt = load(prompt, connection)

    # prepare content
    content = prepare(prompt, inputs)

    # run LLM model
    result = run(prompt, content, configuration, parameters, raw)

    return result
