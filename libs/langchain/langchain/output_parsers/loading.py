from langchain.output_parsers.regex import RegexParser


def load_output_parser(config: dict) -> dict:
    """Load an output parser.

    Args:
        config: config dict

    Returns:
        config dict with output parser loaded
    """
    if "output_parsers" in config and config["output_parsers"] is not None:
        _config = config["output_parsers"]
        output_parser_type = _config["_type"]
        if output_parser_type == "regex_parser":
            output_parser = RegexParser(**_config)
        else:
            msg = f"Unsupported output parser {output_parser_type}"
            raise ValueError(msg)
        config["output_parsers"] = output_parser
    return config
