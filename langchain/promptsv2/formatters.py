from langchain.datasets import SimpleExample, TemplateExample


class SimpleFormatter:
    """A simple formatter that just concatenates the input and output."""

    def apply(self, example: SimpleExample) -> str:
        """Apply the formatter to the example."""
        return f"{example.x_prefix}{example.x}{example.y_prefix}{example.y}{example.stop_sequence}"


class FStringFormatter:
    """Formatter for f-strings."""

    def __init__(self, example_template: str):
        self.example_template = example_template

    def apply(self, example: TemplateExample) -> str:
        return self.example_template.format(**example.inputs)


class MustacheFormatter:
    """Formatter for mustache templates."""

    raise NotImplementedError


DEFAULT_FORMATTER_MAPPING = {
    "simple": SimpleFormatter,
    "f-string": FStringFormatter,
    "mustache": MustacheFormatter,
}
